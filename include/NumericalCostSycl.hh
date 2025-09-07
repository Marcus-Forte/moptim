#pragma once

#include <sycl/sycl.hpp>

#include "ICost.hh"
#include "ILog.hh"
#include "Timer.hh"

static const double g_SyclStep = 1e-9;
constexpr size_t g_localSizeX = 32;
constexpr size_t g_localSizeY = 32;

template <class Model>
class NumericalCostSycl : public ICost {
 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const std::shared_ptr<ILog>& logger, const sycl::queue& queue, const double* input,
                    const double* observations, size_t num_elements, size_t output_dim, size_t param_dim,
                    DifferentiationMethod method = DifferentiationMethod::BACKWARD_EULER)
      : ICost(num_elements),
        logger_(logger),
        queue_(queue),
        input_{input},
        observations_{observations},
        output_dim_{output_dim},
        param_dim_{param_dim},
        method_{method},
        residuals_dim_{num_elements * output_dim} {
    logger_->log(ILog::Level::DEBUG, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());
    logger_->log(ILog::Level::DEBUG, "max_compute_units: {}",
                 queue_.get_device().get_info<sycl::info::device::max_compute_units>());
    logger_->log(ILog::Level::DEBUG, "max_work_group_size: {}",
                 queue_.get_device().get_info<sycl::info::device::max_work_group_size>());
    logger_->log(ILog::Level::DEBUG, "max_work_item_dimensions: {}",
                 queue_.get_device().get_info<sycl::info::device::max_work_item_dimensions>());
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
    // One per column
    residual_plus_data_ = sycl::malloc_device<double>(residuals_dim_ * param_dim_, queue_);
    if (method_ == DifferentiationMethod::CENTRAL) {
      residual_minus_data_ = sycl::malloc_device<double>(residuals_dim_ * param_dim_, queue_);
    }
  }

  ~NumericalCostSycl() override {
    if (!queue_.get_device().is_cpu()) {
      sycl::free(input_sycl_, queue_);
      sycl::free(observations_sycl_, queue_);
    }

    if (method_ == DifferentiationMethod::CENTRAL) {
      sycl::free(residual_minus_data_, queue_);
    }

    sycl::free(cost_reduction_, queue_);
    sycl::free(jacobian_data_, queue_);
    sycl::free(residual_data_, queue_);
    sycl::free(residual_plus_data_, queue_);
  }

  /// \todo if this is called before jacobian, we don't have to compute cost again.
  double computeCost(const Eigen::VectorXd& x) override {
    Model model;
    model.setup(x.data());
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    logger_->log(ILog::Level::DEBUG, "Sycl compute cost items: {}", num_elements_);

    queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction<double>(cost_reduction_, 0.0, sycl::plus<double>{},
                                         sycl::property::reduction::initialize_to_identity{});

      const auto* input_capture = input_sycl_;
      const auto* observations_capture = observations_sycl_;
      const auto output_dim_capture = output_dim_;
      auto* residual_data_capture = residual_data_;

      // const auto workers = sycl::nd_range<1>(sycl::range<1>(num_elements_), sycl::range<1>(1));
      const auto workers = sycl::range<1>(num_elements_);

      cgh.parallel_for(workers, sum, [=](sycl::item<1> id, auto& reduction) {
        const auto ItemRow = id.get_id() * output_dim_capture;
        Eigen::Map<Eigen::VectorXd> residual_map(&residual_data_capture[ItemRow], output_dim_capture);
        model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], &residual_data_capture[ItemRow]);
        reduction += residual_map.squaredNorm();
      });
    });

    queue_.wait();

    double result;
    queue_.copy<double>(cost_reduction_, &result, 1).wait();

    sycl::free(model_sycl, queue_);
    return result;
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override {
    Model model;
    model.setup(x.data());

    if (method_ == ::DifferentiationMethod::BACKWARD_EULER) {
      return applyEulerDiff(x, model);
    }

    return applyCentralDiff(x, model);
  }

 private:
  inline SolveRhs applyEulerDiff(const Eigen::VectorXd& x, Model& model) {
    Timer t0;
    t0.start();
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

    logger_->log(ILog::Level::DEBUG, "Problem space: ({},{}). Group space: ({},{})", num_elements_, param_dim_,
                 g_localSizeX, g_localSizeY);

    auto* JTJ_device = sycl::malloc_device<double>(param_dim_ * param_dim_, queue_);

    auto stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl kernel prepare: took: {} us", stop);

    t0.start();
    // Sycl Kernel /// \todo lots of duplicate code with euler
    queue_.submit([&](sycl::handler& cgh) {
      const auto* input_capture = input_sycl_;
      const auto* observations_capture = observations_sycl_;
      auto* cost_reduction_capture = cost_reduction_;
      auto* residual_data_capture = residual_data_;
      auto* residual_plus_data_capture = residual_plus_data_;
      auto* jacobian_data_capture = jacobian_data_;
      const auto num_elements_capture = num_elements_;
      const auto param_dim_capture = param_dim_;
      const auto output_dim_capture = output_dim_;
      const auto residuals_dim_capture = residuals_dim_;

      auto sum_reduction = sycl::reduction<double>(cost_reduction_, 0.0, sycl::plus<double>{},
                                                   sycl::property::reduction::initialize_to_identity{});

      // Alocate for JTJ reduction per group
      sycl::local_accessor<double, 1> local_JTJ{param_dim_ * param_dim_, cgh};
      sycl::local_accessor<double, 1> local_JT{param_dim_ * output_dim_, cgh};

      const auto ItemsX = ((num_elements_ + g_localSizeX - 1) / g_localSizeX) * g_localSizeX;

      const auto workers = sycl::nd_range<2>{{ItemsX, g_localSizeY}, {g_localSizeX, g_localSizeY}};
      // const auto workers = sycl::range<2>(num_elements_, param_dim_);

      cgh.parallel_for(workers, sum_reduction, [=](sycl::nd_item<2> id, auto& sum_reduction) {
        const auto ItemRow = id.get_global_id(0) * output_dim_capture;
        const auto ItemCol = id.get_global_id(1);

        Eigen::Map<Eigen::VectorXd> residual_map(&residual_data_capture[ItemRow], output_dim_capture);
        Eigen::Map<Eigen::VectorXd> residual_plus_map(
            &residual_plus_data_capture[ItemRow + ItemCol * residuals_dim_capture], output_dim_capture);
        Eigen::Map<Eigen::MatrixXd> jacobian_map(jacobian_data_capture, residuals_dim_capture, param_dim_capture);

        Eigen::Map<Eigen::MatrixXd> local_JTJ_map(&local_JTJ[0], param_dim_capture, param_dim_capture);
        Eigen::Map<Eigen::MatrixXd> local_JT_map(&local_JT[0], param_dim_capture, param_dim_capture);

        if (ItemRow < residuals_dim_capture && ItemCol < param_dim_capture) {
          models_sycl_plus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow],
                                      residual_plus_map.data());

          if (ItemCol == 0) {
            model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_map.data());
            sum_reduction += residual_map.squaredNorm();
          }
        }

        sycl::group_barrier(id.get_group());

        if (ItemRow < residuals_dim_capture && ItemCol < param_dim_capture) {
          jacobian_map.block(ItemRow, ItemCol, output_dim_capture, 1) = (residual_plus_map - residual_map) / g_SyclStep;
        }

        /// \todo synchronize threads and compute transpose

        // local_JTJ_map.col(ItemCol) = (residual_plus_map - residual_map) / g_SyclStep;

        // sycl::group_barrier(id.get_group());
        // // Leaders (as many as num elements)
        // if (id.get_local_id(1) == 0) {
        //   auto* pt = &local_sum[0];
        //   Eigen::Map<Eigen::MatrixXd> jtj_reduction(JTJ_device, param_dim_capture, param_dim_capture);
        //   local_JT_map.noalias() = local_JTJ_map.transpose();
        //   local_JTJ_map.noalias() = local_JT_map * local_JTJ_map;

        //   jtj_reduction.noalias() = jtj_reduction.transpose() * jtj_reduction;
        //   local_sum_map.setOnes();
        //   // jtj_reduction = local_sum_map;
        //   out << "Reduce\n";
        //   jtj_reduce_.combine(local_sum_map);
        // }
      });
    });

    // Eigen::MatrixXd JTJ_result(param_dim_, param_dim_);
    // queue_.copy<double>(JTJ_device, JTJ_result.data(), JTJ_result.size()).wait();

    /// \todo compute those inside kernel
    Eigen::MatrixXd Jac(residuals_dim_, param_dim_);
    Eigen::VectorXd Err(residuals_dim_);

    double Sum;

    queue_.wait();
    stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl kernel jacobian: took: {} us", stop);
    t0.start();
    queue_.copy<double>(jacobian_data_, Jac.data(), Jac.size());
    queue_.copy<double>(residual_data_, Err.data(), Err.size());
    queue_.copy<double>(cost_reduction_, &Sum, 1);
    stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl result data copy took: {} us", stop);

    t0.start();
    const auto&& JTJ = (Jac.transpose() * Jac).eval();
    const auto&& JTB = (Jac.transpose() * Err).eval();
    stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl jacobian result: {} us", stop);

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);

    return {JTJ, JTB, Sum};
  }

  /// \todo fixme
  inline SolveRhs applyCentralDiff(const Eigen::VectorXd& x, Model& model) {
    // Sycl captures
    const auto* input_capture = input_sycl_;
    const auto* observations_capture = observations_sycl_;
    auto* cost_reduction_capture = cost_reduction_;
    auto* residual_data_capture = residual_data_;
    auto* residual_plus_data_capture = residual_plus_data_;
    auto* residual_minus_data_capture = residual_minus_data_;
    auto* jacobian_data_capture = jacobian_data_;

    const auto param_dim_capture = param_dim_;
    const auto output_dim_capture = output_dim_;
    const auto residuals_dim_capture = residuals_dim_;
    //

    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(param_dim_);
    std::vector<std::shared_ptr<Model>> models_minus(param_dim_);

    auto* models_sycl_plus = sycl::malloc_device<Model>(param_dim_, queue_);
    auto* models_sycl_minus = sycl::malloc_device<Model>(param_dim_, queue_);
    for (int i = 0; i < param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      Eigen::VectorXd x_minus(x);
      x_plus[i] += g_SyclStep;
      x_minus[i] -= g_SyclStep;

      models_plus[i] = std::make_shared<Model>();
      models_plus[i]->setup(x_plus.data());
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1);

      models_minus[i] = std::make_shared<Model>();
      models_minus[i]->setup(x_minus.data());
      queue_.copy<Model>(models_minus[i].get(), &models_sycl_minus[i], 1);
    }

    // Sycl Kernel
    queue_.submit([&](sycl::handler& cgh) {
      const auto sum = sycl::reduction(cost_reduction_capture, 0.0, sycl::plus<double>{},
                                       sycl::property::reduction::initialize_to_identity{});

      const auto workers = sycl::range<2>(num_elements_, param_dim_);

      cgh.parallel_for(workers, sum, [=](sycl::item<2> id, auto& reduction) {
        const auto ItemRow = id[0] * output_dim_capture;
        const auto ItemCol = id[1];

        Eigen::Map<Eigen::VectorXd> residual_map(&residual_data_capture[ItemRow], output_dim_capture);
        Eigen::Map<Eigen::VectorXd> residual_plus_map(
            &residual_plus_data_capture[ItemRow + ItemCol * residuals_dim_capture], output_dim_capture);
        Eigen::Map<Eigen::VectorXd> residual_minus_map(
            &residual_minus_data_capture[ItemRow + ItemCol * residuals_dim_capture], output_dim_capture);
        Eigen::Map<Eigen::MatrixXd> jacobian_map(jacobian_data_capture, residuals_dim_capture, param_dim_capture);

        /// \todo Only need to compute those once per row
        model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_map.data());

        models_sycl_plus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_plus_map.data());
        models_sycl_minus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow],
                                     residual_minus_map.data());
        jacobian_map.block(ItemRow, ItemCol, output_dim_capture, 1) =
            (residual_plus_map - residual_minus_map) / (2 * g_SyclStep);

        // Only compute `InputSize` (One Column) times.
        if (ItemCol == 0) {
          Eigen::Map<Eigen::VectorXd> residual_data_map(residual_data_capture, residuals_dim_capture);
          reduction += residual_data_map.block(ItemRow, 0, output_dim_capture, 1).squaredNorm();
        }

        /// \todo synchronize threads and compute transpose
      });
    });

    /// \todo compute those inside kernel
    Eigen::MatrixXd Jac(residuals_dim_, param_dim_capture);
    Eigen::VectorXd Err(residuals_dim_);
    double Sum;

    queue_.wait();
    queue_.copy<double>(jacobian_data_, Jac.data(), Jac.size()).wait();
    queue_.copy<double>(residual_data_, Err.data(), Err.size()).wait();
    queue_.copy<double>(cost_reduction_, &Sum, 1).wait();

    const auto&& JTJ = Jac.transpose() * Jac;
    const auto&& JTB = Jac.transpose() * Err;

    sycl::free(models_sycl_plus, queue_);
    sycl::free(models_sycl_minus, queue_);
    sycl::free(model_sycl, queue_);

    return {JTJ, JTB, Sum};
  }

  std::shared_ptr<ILog> logger_;

  const double* input_;
  const double* observations_;
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