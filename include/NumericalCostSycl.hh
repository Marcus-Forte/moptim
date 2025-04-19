#pragma once

#include "ConsoleLogger.hh"
#include "ICost.hh"
#include "sycl/sycl.hpp"

static const double g_SyclStep = 1e-1;

template <class Model>
class NumericalCostSycl : public ICost {
 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const sycl::queue& queue, const double* input, const double* observations, size_t num_elements,
                    size_t output_dim, size_t param_dim,
                    DifferentiationMethod method = DifferentiationMethod::BACKWARD_EULER)
      : queue_(queue),
        input_{input},
        observations_{observations},
        num_elements_{num_elements},
        output_dim_{output_dim},
        param_dim_{param_dim},
        method_{method},
        residuals_dim_{num_elements * output_dim} {
    ConsoleLogger logger;
    logger.log(ILog::Level::INFO, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());

    if (!queue_.get_device().is_cpu()) {
      input_sycl_ = sycl::malloc_device<double>(residuals_dim_, queue_);
      observations_sycl_ = sycl::malloc_device<double>(residuals_dim_, queue_);
      queue_.copy<double>(input, input_sycl_, residuals_dim_);
      queue_.copy<double>(observations, observations_sycl_, residuals_dim_);

    } else {  // No need to copy data if it already lies in a CPU (host) device
      input_sycl_ = const_cast<double*>(input);
      observations_sycl_ = const_cast<double*>(observations);
    }

    cost_reduction_ = sycl::malloc_device<double>(1, queue_);
    jacobian_data_ = sycl::malloc_device<double>(residuals_dim_ * param_dim_, queue_);
    residual_data_ = sycl::malloc_device<double>(residuals_dim_, queue_);
    residual_plus_data_ = sycl::malloc_device<double>(residuals_dim_, queue_);
  }

  virtual ~NumericalCostSycl() {
    if (!queue_.get_device().is_cpu()) {
      sycl::free(input_sycl_, queue_);
      sycl::free(observations_sycl_, queue_);
    }

    sycl::free(cost_reduction_, queue_);
    sycl::free(jacobian_data_, queue_);
    sycl::free(residual_data_, queue_);
    sycl::free(residual_plus_data_, queue_);
  }

  double computeCost(const Eigen::VectorXd& x) override {
    const auto* input_capture = input_sycl_;
    const auto* observations_capture = observations_sycl_;
    const auto output_dim_capture = output_dim_;
    auto* cost_reduction_capture = cost_reduction_;
    auto* residual_data_capture = residual_data_;

    Model model;
    model.setup(x.data());
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction<double>(cost_reduction_capture, 0.0, sycl::plus<double>{},
                                         sycl::property::reduction::initialize_to_identity{});

      cgh.parallel_for(sycl::range<1>(num_elements_), sum, [=](sycl::id<1> id, auto& reduction) {
        const auto ItemRow = id[0] * output_dim_capture;

        model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], &residual_data_capture[ItemRow]);
        double norm = 0.0;
        for (int i = 0; i < output_dim_capture; ++i) {
          norm += residual_data_capture[ItemRow + i] * residual_data_capture[ItemRow + i];
        }

        reduction += norm;
      });
    });

    queue_.wait();

    double result;
    queue_.copy<double>(cost_reduction_, &result, 1).wait();

    sycl::free(model_sycl, queue_);
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
    auto* cost_reduction_capture = cost_reduction_;
    auto* residual_data_capture = residual_data_;
    auto* residual_plus_data_capture = residual_plus_data_;
    auto* jacobian_data_capture = jacobian_data_;

    const auto param_dim_capture = param_dim_;
    const auto output_dim_capture = output_dim_;
    const auto residuals_dim_capture = residuals_dim_;

    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(param_dim_);
    auto* models_sycl_plus = sycl::malloc_device<Model>(param_dim_, queue_);
    for (int i = 0; i < param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_SyclStep;
      models_plus[i] = std::make_shared<Model>();
      models_plus[i]->setup(x_plus.data());
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1);
    }

    queue_.submit([&](sycl::handler& cgh) {
      const auto sum = sycl::reduction(cost_reduction_capture, 0.0, sycl::plus<double>{},
                                       sycl::property::reduction::initialize_to_identity{});

      const auto workers = sycl::range<2>(num_elements_, param_dim_);

      cgh.parallel_for(workers, sum, [=](sycl::item<2> id, auto& reduction) {
        const auto ItemRow = id[0] * output_dim_capture;
        const auto ItemCol = id[1];

        Eigen::Map<Eigen::VectorXd> residual_map(&residual_data_capture[ItemRow], output_dim_capture);
        Eigen::Map<Eigen::VectorXd> residual_plus_map(&residual_plus_data_capture[ItemRow], output_dim_capture);
        Eigen::Map<Eigen::MatrixXd> jacobian_map(jacobian_data_capture, residuals_dim_capture, param_dim_capture);

        /// \todo Only need to compute those once per row
        model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_map.data());
        models_sycl_plus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_plus_map.data());

        // jacobian_map.block(ItemRow, ItemCol, output_dim_capture, 1) = (residual_plus_map - residual_map) /
        // g_SyclStep;

        // residual_map[0] = static_cast<double>(id.get_linear_id());
        // residual_map[0] = 1.0;
        // residual_plus_map[0] = 4.0;

        // residual_map[1] = 1.0;
        residual_plus_map[1] = 3.0;
        // cast[1] = 22;

        // jacobian_map(ItemRow, ItemCol) = id.get_linear_id();
        jacobian_map.block(ItemRow, ItemCol, output_dim_capture, 1) = (residual_plus_map - residual_map);
        // Only compute `InputSize` (One Column) times.
        if (ItemCol == 0) {
          Eigen::Map<Eigen::VectorXd> residual_data_map(residual_data_capture, residuals_dim_capture);
          reduction += residual_data_map.block(ItemRow, 0, output_dim_capture, 1).squaredNorm();
        }
      });
    });

    queue_.wait();

    /// \todo compute those inside kernel
    Eigen::MatrixXd Jac(residuals_dim_, param_dim_capture);
    Eigen::VectorXd Err(residuals_dim_);

    queue_.copy<double>(jacobian_data_, Jac.data(), Jac.size()).wait();
    queue_.copy<double>(residual_data_, Err.data(), Err.size()).wait();
    queue_.copy<double>(cost_reduction_, &std::get<2>(init), 1).wait();

    std::get<0>(init) = Jac.transpose() * Jac;
    std::get<1>(init) = Jac.transpose() * Err;

    std::cout << "Jacobian: " << Jac << std::endl;

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);

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
  const size_t num_elements_;
  const size_t output_dim_;
  const size_t residuals_dim_;
  const size_t param_dim_;

  const DifferentiationMethod method_;

  double* input_sycl_;
  double* observations_sycl_;
  double* cost_reduction_;
  double* jacobian_data_;
  double* residual_data_;
  double* residual_plus_data_;
  double* residual_minus_data_;
  sycl::queue queue_;
};