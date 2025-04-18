#pragma once

#include "ConsoleLogger.hh"
#include "ICost.hh"
#include "sycl/sycl.hpp"

static const double g_SyclStep = 1e-9;

template <class Model, int OutputDim>
class NumericalCostSycl : public ICost {
  using ResidualVectorT = Eigen::Vector<double, OutputDim>;

 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const sycl::queue& queue, const double* input, const double* observations, size_t input_size,
                    DifferentiationMethod method = DifferentiationMethod::BACKWARD_EULER)
      : queue_(queue), input_{input}, observations_{observations}, input_size_{input_size}, method_{method} {
    ConsoleLogger logger;
    logger.log(ILog::Level::INFO, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());

    if (!queue_.get_device().is_cpu()) {
      input_sycl_ = sycl::malloc_device<double>(input_size * OutputDim, queue_);
      observations_sycl_ = sycl::malloc_device<double>(input_size * OutputDim, queue_);
      queue_.copy<double>(input, input_sycl_, input_size * OutputDim);
      queue_.copy<double>(observations, observations_sycl_, input_size * OutputDim);

    } else {  // No need to copy data if it already lies in a CPU (host) device
      input_sycl_ = const_cast<double*>(input);
      observations_sycl_ = const_cast<double*>(observations);
    }
  }

  virtual ~NumericalCostSycl() {
    if (!queue_.get_device().is_cpu()) {
      sycl::free(input_sycl_, queue_);
      sycl::free(observations_sycl_, queue_);
    }
  }

  double computeCost(const Eigen::VectorXd& x) override {
    const auto* input_capture = input_sycl_;
    const auto* observations_capture = observations_sycl_;
    const auto input_size_capture = input_size_;

    Model model;
    model.setup(x.data());
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    queue_.memset(cost_reduction, 0, sizeof(double));

    queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

      cgh.parallel_for(sycl::range<1>(input_size_capture), sum, [=](sycl::id<1> id, auto& reduction) {
        double residual[OutputDim];
        model_sycl->f(&input_capture[id], &observations_capture[id], residual);
        double norm = 0.0;
        for (int dim = 0; dim < OutputDim; ++dim) {
          norm += residual[dim] * residual[dim];
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

    Model model;
    model.setup(x.data());

    if (method_ == ::DifferentiationMethod::BACKWARD_EULER) {
      return applyEulerDiff(x, model, init);
    }

    // return applyCentralDiff(x, model, init);

    return init;
  }

 private:
  inline SolveRhs applyEulerDiff(const Eigen::VectorXd& x, Model& model, SolveRhs& init) {
    // Sycl captures
    const auto* input_capture = input_sycl_;
    const auto* observations_capture = observations_sycl_;
    const auto param_dim_capture = x.size();
    const auto input_size_capture = input_size_;

    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(param_dim_capture);
    auto* models_sycl_plus = sycl::malloc_device<Model>(param_dim_capture, queue_);
    for (int i = 0; i < param_dim_capture; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_SyclStep;
      models_plus[i] = std::make_shared<Model>();
      models_plus[i]->setup(x_plus.data());
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1);
    }

    // auto* cost_reduction = sycl::malloc_device<double>(1, queue_);
    // queue_.memset(cost_reduction, 0, sizeof(double));

    auto* jacobian_data = sycl::malloc_device<double>(OutputDim * input_size_capture * param_dim_capture, queue_);
    auto* residual_data = sycl::malloc_device<double>(OutputDim * input_size_capture, queue_);

    queue_.submit([&](sycl::handler& cgh) {
      const auto workers = sycl::range<2>(input_size_capture, param_dim_capture);

      cgh.parallel_for(workers, [=](sycl::item<2> id) {
        const auto ItemRow = id[0];
        const auto ItemCol = id[1];

        double residual[OutputDim];
        double residual_plus[OutputDim];
        model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], residual);
        models_sycl_plus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_plus);

        Eigen::Map<Eigen::VectorXd> residual_map(residual, OutputDim);
        Eigen::Map<Eigen::VectorXd> residual_plus_map(residual_plus, OutputDim);
        Eigen::Map<Eigen::VectorXd> jacobian_map(jacobian_data, OutputDim * input_size_capture, param_dim_capture);
        jacobian_map.block<OutputDim, 1>(ItemRow * OutputDim, ItemCol) =
            (residual_plus_map - residual_map) / g_SyclStep;
        // Only compute `InputSize` (One Column) times.
        if (ItemCol == 0) {
          Eigen::Map<Eigen::VectorXd> residual_data_map(residual_data, OutputDim * input_size_capture);
          residual_data_map.block<OutputDim, 1>(ItemRow * OutputDim, 0) = residual_map;
        }
      });
    });

    queue_.wait();

    /// \todo compute those inside kernel
    Eigen::MatrixXd Jac(OutputDim * input_size_capture, param_dim_capture);
    Eigen::VectorXd Err(OutputDim * input_size_capture);

    queue_.copy<double>(jacobian_data, Jac.data(), Jac.size()).wait();
    queue_.copy<double>(residual_data, Err.data(), Err.size()).wait();
    // queue_.copy<double>(cost_reduction, &std::get<2>(init), 1).wait();

    std::cout << "sycl:" << Jac << std::endl;

    std::get<0>(init) = Jac.transpose() * Jac;
    std::get<1>(init) = Jac.transpose() * Err;
    std::get<2>(init) = Err.squaredNorm();

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);
    sycl::free(jacobian_data, queue_);
    sycl::free(residual_data, queue_);

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

  const double* input_;
  const double* observations_;
  const size_t input_size_;
  double* input_sycl_;
  double* observations_sycl_;
  const DifferentiationMethod method_;

  sycl::queue queue_;
};