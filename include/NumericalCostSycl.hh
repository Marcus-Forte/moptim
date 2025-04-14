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
  NumericalCostSycl(const std::vector<InputT>* input, const std::vector<OutputT>* observations,
                    const sycl::queue& queue)
      : input_{input}, observations_{observations}, queue_(queue) {
    ConsoleLogger logger;
    logger.log(ILog::Level::INFO, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());

    if (!queue_.get_device().is_cpu()) {
      input_sycl_ = sycl::malloc_device<InputT>(input->size(), queue_);
      observations_sycl_ = sycl::malloc_device<OutputT>(observations->size(), queue_);
      queue_.copy<InputT>(input->data(), input_sycl_, input->size());
      queue_.copy<OutputT>(observations->data(), observations_sycl_, observations->size());

    } else {  // No need to copy data if it already lies in a CPU (host) device
      input_sycl_ = const_cast<InputT*>(input->data());
      observations_sycl_ = const_cast<OutputT*>(observations->data());
    }
  }

  virtual ~NumericalCostSycl() {
    if (!queue_.get_device().is_cpu()) {
      sycl::free(input_sycl_, queue_);
      sycl::free(observations_sycl_, queue_);
    }
  }

  double computeCost(const Eigen::VectorXd& x) override {
    const auto input_capture = input_sycl_;
    const auto observations_capture = observations_sycl_;
    const auto input_size_capture = input_->size();

    Model model(x);
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    queue_.memset(cost_reduction, 0, sizeof(double));

    queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

      cgh.parallel_for(sycl::range<1>(input_size_capture), sum, [=](sycl::id<1> id, auto& reduction) {
        const OutputT&& residual = (*model_sycl)(input_capture[id], observations_capture[id]);

        const auto* error_presidual_ptr(reinterpret_cast<const double*>(&residual));
        double norm = 0.0;
        for (int dim = 0; dim < OutputDim; ++dim) {
          norm += error_presidual_ptr[dim] * error_presidual_ptr[dim];
        }

        reduction += norm;
      });
    });

    queue_.wait();

    double result;
    queue_.copy<double>(cost_reduction, &result, 1).wait();

    sycl::free(model_sycl, queue_);
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
    // Sycl captures
    const auto input_capture = input_sycl_;
    const auto observations_capture = observations_sycl_;
    const auto param_dim_capture = x.size();
    const auto input_size_capture = input_->size();

    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(param_dim_capture);
    auto* models_sycl_plus = sycl::malloc_device<Model>(param_dim_capture, queue_);
    for (int i = 0; i < param_dim_capture; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_SyclStep;
      models_plus[i] = std::make_shared<Model>(x_plus);
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1);
    }

    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    queue_.memset(cost_reduction, 0, sizeof(double));

    auto* jacobian_data = sycl::malloc_device<double>(OutputDim * input_size_capture * param_dim_capture, queue_);
    auto* error_data = sycl::malloc_device<double>(OutputDim * input_size_capture, queue_);

    queue_.wait();

    queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

      const auto workers = sycl::nd_range<2>(sycl::range<2>(input_size_capture, param_dim_capture),
                                             sycl::range<2>(OutputDim, param_dim_capture));
      // const auto workers = sycl::range<2>(InputSize, ParamDim);

      cgh.parallel_for(workers, sum, [=](sycl::nd_item<2> id, auto& reduction) {
        const auto ItemRow = id.get_global_id(0);
        const auto ItemCol = id.get_global_id(1);

        const OutputT&& residual = model_sycl[0](input_capture[ItemRow], observations_capture[ItemRow]);
        const auto numRows = OutputDim * input_size_capture;
        auto* jacobian_col = reinterpret_cast<OutputT*>(&jacobian_data[ItemCol * numRows + ItemRow * OutputDim]);
        *jacobian_col =
            ((models_sycl_plus[ItemCol](input_capture[ItemRow], observations_capture[ItemRow])) - residual) /
            g_SyclStep;

        // Only compute `InputSize` (One Column) times.
        if (ItemCol == 0) {
          (*reinterpret_cast<OutputT*>(&error_data[ItemRow * OutputDim])) = residual;

          const auto* error_ptr(reinterpret_cast<const double*>(&residual));
          double norm = 0.0;
          for (int dim = 0; dim < OutputDim; ++dim) {
            norm += error_ptr[dim] * error_ptr[dim];
          }

          reduction += norm;
        }
      });
    });

    queue_.wait();

    /// \todo compute those inside kernel
    Eigen::MatrixXd Jac(OutputDim * input_size_capture, param_dim_capture);
    Eigen::VectorXd Err(OutputDim * input_size_capture);

    queue_.copy<double>(jacobian_data, Jac.data(), Jac.size()).wait();
    queue_.copy<double>(error_data, Err.data(), Err.size()).wait();
    queue_.copy<double>(cost_reduction, &std::get<2>(init), 1).wait();

    std::get<0>(init) = Jac.transpose() * Jac;
    std::get<1>(init) = Jac.transpose() * Err;

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);
    sycl::free(jacobian_data, queue_);
    sycl::free(error_data, queue_);
    sycl::free(cost_reduction, queue_);

    return init;
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
  sycl::queue queue_;
};