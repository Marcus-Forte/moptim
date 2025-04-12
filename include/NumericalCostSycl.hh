#pragma once

#include "ConsoleLogger.hh"
#include "ICost.hh"
#include "sycl/sycl.hpp"

static const double g_SyclStep = 1e-12;

template <class InputT, class OutputT, class Model,
          ::DifferentiationMethod MethodT = ::DifferentiationMethod::BACKWARD_EULER>
class NumericalCostSycl : public ICost {
  static constexpr size_t OutputDim = sizeof(OutputT) / sizeof(double);
  using ResidualVectorT = Eigen::Vector<double, OutputDim>;

 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const std::vector<InputT>* input, const std::vector<OutputT>* observations)
      : input_{input}, observations_{observations}, queue_(sycl::default_selector_v) {
    /// \todo if device is CPU, no need to copy.
    input_sycl_ = sycl::malloc_device<InputT>(input->size(), queue_);
    observations_sycl_ = sycl::malloc_device<OutputT>(observations->size(), queue_);
    ConsoleLogger logger;
    logger.log(ILog::Level::INFO, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());

    /// \todo Copy data to GPU now?
    queue_.copy<InputT>(input->data(), input_sycl_, input->size()).wait();
    queue_.copy<OutputT>(observations->data(), observations_sycl_, observations->size()).wait();
  }

  virtual ~NumericalCostSycl() {
    sycl::free(input_sycl_, queue_);
    sycl::free(observations_sycl_, queue_);
  }

  double computeCost(const Eigen::VectorXd& x) override {
    auto input_capture = input_sycl_;
    auto observations_capture = observations_sycl_;

    Model model(x);
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1).wait();

    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    queue_.single_task([=]() { *cost_reduction = 0.0F; }).wait();

    queue_
        .submit([&](sycl::handler& cgh) {
          auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

          cgh.parallel_for(sycl::range<1>(input_->size()), sum, [=](sycl::id<1> id, auto& reduction) {
            auto error = (*model_sycl)(input_capture[id], observations_capture[id]);

            Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&error));
            reduction += residual_map.squaredNorm();
          });
        })
        .wait();

    double result;
    queue_.copy<double>(cost_reduction, &result, 1).wait();
    sycl::free(cost_reduction, queue_);
    return result;
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override {
    SolveRhs init{Eigen::MatrixXd::Zero(x.size(), x.size()), Eigen::VectorXd::Zero(x.size()), 0.0};

    Model model(x);

    if constexpr (MethodT == ::DifferentiationMethod::BACKWARD_EULER) {
      return applyEulerDiff(x, model, init);
    } else {
      // return applyCentralDiff(x, model, init);
    }
  }

 private:
  inline SolveRhs applyEulerDiff(const Eigen::VectorXd& x, Model& model, SolveRhs& init) {
    auto input_capture = input_sycl_;
    auto observations_capture = observations_sycl_;
    const auto ParamDim = x.size();
    const auto InputSize = input_->size();

    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1).wait();

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(ParamDim);
    auto* models_sycl_plus = sycl::malloc_device<Model>(ParamDim, queue_);
    for (int i = 0; i < ParamDim; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_SyclStep;
      models_plus[i] = std::make_shared<Model>(x_plus);
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1).wait();
    }

    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    queue_.memset(cost_reduction, 0, sizeof(double)).wait();

    // Allocate enough memory to compute the results

    auto* JacobianData = sycl::malloc_device<double>(OutputDim * InputSize * ParamDim, queue_);
    auto* ErrorData = sycl::malloc_device<double>(OutputDim * InputSize, queue_);

    /// \todo either: Reduce in kernel or compute matrix transposes in device.
    queue_
        .submit([&](sycl::handler& cgh) {
          auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

          const auto workers =
              sycl::nd_range<2>(sycl::range<2>(InputSize, ParamDim), sycl::range<2>(OutputDim, ParamDim));
          // const auto workers = sycl::range<2>(InputSize, ParamDim);

          cgh.parallel_for(workers, sum, [=](sycl::nd_item<2> id, auto& reduction) {
            const auto ItemRow = id.get_global_id(0);
            const auto ItemCol = id.get_global_id(1);

            const auto&& residual = model_sycl[0](input_capture[ItemRow], observations_capture[ItemRow]);
            const auto numRows = OutputDim * InputSize;
            auto* jacobian_col = reinterpret_cast<OutputT*>(&JacobianData[ItemCol * numRows + ItemRow * OutputDim]);
            *jacobian_col =
                ((models_sycl_plus[ItemCol](input_capture[ItemRow], observations_capture[ItemRow])) - residual) /
                g_SyclStep;

            // Only compute `InputSize` times.
            if (ItemCol == 0) {
              (*reinterpret_cast<OutputT*>(&ErrorData[ItemRow * OutputDim])) = residual;
              Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
              reduction += residual_map.squaredNorm();
            }

            // Synchronize Jacobian transpose calculation per row
            // id.barrier(sycl::access::fence_space::local_space);
            // Assign of the two in the group to calculate

            // Eigen::Map<Eigen::Matrix<double, -1, -1>> JacobianMap(JacobianData[], OutputDim, ParamDim);
          });
        })
        .wait();

    Eigen::MatrixXd Jac(OutputDim * InputSize, ParamDim);
    Eigen::VectorXd Err(OutputDim * InputSize);

    queue_.copy<double>(JacobianData, Jac.data(), Jac.size()).wait();
    queue_.copy<double>(ErrorData, Err.data(), Err.size()).wait();

    std::get<0>(init) = Jac.transpose() * Jac;
    // std::get<0>(init).resize(OutputDim * InputSize, ParamDim);
    // queue_.copy<double>(JacobianData, std::get<0>(init).data(), Jac.size()).wait();
    std::get<1>(init) = Jac.transpose() * Err;
    queue_.copy<double>(cost_reduction, &std::get<2>(init), 1).wait();

    // std::cout << std::get<0>(init) << std::endl;

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);
    sycl::free(JacobianData, queue_);
    sycl::free(cost_reduction, queue_);

    return init;

    // const auto&& jacobian = [&](const InputT& input, const OutputT& observation) -> SolveRhs {
    //   Eigen::MatrixXd&& jacobian_matrix{OutputDim, x.size()};

    //   const auto&& residual = model(input, observation);
    //   for (int i = 0; i < x.size(); ++i) {
    //     auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_matrix.col(i).data());
    //     *jacobian_col = ((*models_plus[i])(input, observation) - residual) / g_step;
    //   }

    //   Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
    //   const auto&& JTJ = jacobian_matrix.transpose() * jacobian_matrix;
    //   const auto&& JTb = jacobian_matrix.transpose() * residual_map;
    //   return {JTJ, JTb, residual_map.squaredNorm()};
    // };

    // return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
    //                              ICost::Reduction, jacobian);
  }

  // inline SolveRhs applyCentralDiff(const Eigen::VectorXd& x, Model& model, SolveRhs& init) {
  //   // Initialize vector of models
  //   std::vector<std::shared_ptr<Model>> models_plus(x.size());
  //   std::vector<std::shared_ptr<Model>> models_minus(x.size());

  //   for (int i = 0; i < x.size(); ++i) {
  //     Eigen::VectorXd x_plus(x);
  //     Eigen::VectorXd x_minus(x);
  //     x_plus[i] += g_step;
  //     x_minus[i] -= g_step;
  //     models_plus[i] = std::make_shared<Model>(x_plus);
  //     models_minus[i] = std::make_shared<Model>(x_minus);
  //   }

  //   const auto jacobian = [&](const InputT& input, const OutputT& observation) -> SolveRhs {
  //     Eigen::MatrixXd&& jacobian_matrix{OutputDim, x.size()};

  //     const auto&& residual = model(input, observation);
  //     for (int i = 0; i < x.size(); ++i) {
  //       auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_matrix.col(i).data());
  //       *jacobian_col = ((*models_plus[i])(input, observation) - (*models_minus[i])(input, observation)) / (2 *
  //       g_step);
  //     }

  //     Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
  //     const auto&& JTJ = jacobian_matrix.transpose() * jacobian_matrix;
  //     const auto&& JTb = jacobian_matrix.transpose() * residual_map;
  //     return {JTJ, JTb, residual_map.squaredNorm()};
  //   };

  //   return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
  //                                ICost::Reduction, jacobian);
  // }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;

  InputT* input_sycl_;
  OutputT* observations_sycl_;
  //   OutputT* output_sycl_;
  sycl::queue queue_;
};