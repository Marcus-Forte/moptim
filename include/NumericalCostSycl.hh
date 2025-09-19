#pragma once

#include <Eigen/Dense>
#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

#include "ICost.hh"
#include "ILog.hh"
#include "Timer.hh"

namespace moptim {
template <class T, class Model, oneapi::math::backend Backend = oneapi::math::backend::generic>
class NumericalCostSycl : public ICost<T> {
 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const std::shared_ptr<ILog>& logger, const sycl::queue& queue, const T* input,
                    const T* observations, size_t input_dim, size_t observation_dim, size_t param_dim,
                    size_t num_elements)
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
        logger_(logger),
        queue_(queue),
        input_{input},
        observations_{observations} {
    logger_->log(ILog::Level::DEBUG, "Sycl Device: {}", queue_.get_device().get_info<sycl::info::device::name>());
    logger_->log(ILog::Level::DEBUG, "max_compute_units: {}",
                 queue_.get_device().get_info<sycl::info::device::max_compute_units>());
    logger_->log(ILog::Level::DEBUG, "max_work_group_size: {}",
                 queue_.get_device().get_info<sycl::info::device::max_work_group_size>());
    logger_->log(ILog::Level::DEBUG, "max_work_item_dimensions: {}",
                 queue_.get_device().get_info<sycl::info::device::max_work_item_dimensions>());

    logger_->log(ILog::Level::DEBUG, "param_dim: {}", param_dim);

    if (!queue_.get_device().is_cpu()) {
      input_sycl_ = sycl::malloc_device<T>(observation_dim_ * num_elements, queue_);
      observations_sycl_ = sycl::malloc_device<T>(observation_dim_ * num_elements, queue_);

      queue_.copy<T>(input, input_sycl_, observation_dim_ * num_elements);
      queue_.copy<T>(observations, observations_sycl_, observation_dim_ * num_elements);

    } else {  // No need to copy data if it already lies in a CPU (host) device
      input_sycl_ = const_cast<T*>(input);
      observations_sycl_ = const_cast<T*>(observations);
    }

    cost_reduction_ = sycl::malloc_device<T>(1, queue_);
    jacobian_data_ = sycl::malloc_device<T>(observation_dim_ * num_elements * param_dim_, queue_);
    residual_data_ = sycl::malloc_device<T>(observation_dim_ * num_elements, queue_);
    // One per column
    residual_plus_data_ = sycl::malloc_device<T>(observation_dim_ * num_elements * param_dim_, queue_);
  }

  ~NumericalCostSycl() override {
    if (!queue_.get_device().is_cpu()) {
      sycl::free(input_sycl_, queue_);
      sycl::free(observations_sycl_, queue_);
    }

    sycl::free(cost_reduction_, queue_);
    sycl::free(jacobian_data_, queue_);
    sycl::free(residual_data_, queue_);
    sycl::free(residual_plus_data_, queue_);
  }

  /// \todo if this is called before jacobian, we don't have to compute cost again.
  T computeCost(const T* x) override {
    Model model;
    model.setup(x);
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    const auto copy_model_event = queue_.copy<Model>(&model, model_sycl, 1);

    logger_->log(ILog::Level::DEBUG, "Sycl compute cost items: {}", num_elements_);

    queue_
        .submit([&](sycl::handler& cgh) {
          cgh.depends_on(copy_model_event);
          auto sum = sycl::reduction<T>(cost_reduction_, 0.0, sycl::plus<T>{},
                                        sycl::property::reduction::initialize_to_identity{});

          const auto* input_capture = input_sycl_;
          const auto* observations_capture = observations_sycl_;
          const auto observation_dim_capture = observation_dim_;
          auto* residual_data_capture = residual_data_;

          const auto workers = sycl::range<1>(num_elements_);

          cgh.parallel_for(workers, sum, [=](sycl::item<1> id, auto& reduction) {
            const auto ItemRow = id.get_id() * observation_dim_capture;
            Eigen::Map<VectorT> residual_map(&residual_data_capture[ItemRow], observation_dim_capture);
            model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], &residual_data_capture[ItemRow]);
            reduction += residual_map.squaredNorm();
          });
        })
        .wait();

    T result;
    queue_.copy<T>(cost_reduction_, &result, 1).wait();

    sycl::free(model_sycl, queue_);
    return result;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) override {
    Eigen::Map<const VectorT> x_vec(x, param_dim_);

    Model model;
    model.setup(x);

    Timer t0;
    t0.start();
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1);

    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(param_dim_);
    auto* models_sycl_plus = sycl::malloc_device<Model>(param_dim_, queue_);
    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());

    for (int i = 0; i < param_dim_; ++i) {
      VectorT x_plus(x_vec);
      x_plus[i] += g_step;
      models_plus[i] = std::make_shared<Model>();
      models_plus[i]->setup(x_plus.data());
      queue_.copy<Model>(models_plus[i].get(), &models_sycl_plus[i], 1);
    }

    queue_.wait();

    auto stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl kernel prepare: took: {} us", stop);

    t0.start();

    // // Sycl Kernel /// \todo lots of duplicate code with euler
    auto jac_event = queue_.submit([&](sycl::handler& cgh) {
      const auto* input_capture = input_sycl_;
      const auto* observations_capture = observations_sycl_;
      auto* cost_reduction_capture = cost_reduction_;
      auto* residual_data_capture = residual_data_;
      auto* residual_plus_data_capture = residual_plus_data_;
      auto* jacobian_data_capture = jacobian_data_;
      const auto num_elements_capture = num_elements_;
      const auto param_dim_capture = param_dim_;
      const auto output_dim_capture = observation_dim_;
      const auto residuals_dim_capture = observation_dim_ * num_elements_;

      auto sum_reduction = sycl::reduction<T>(cost_reduction_, 0.0, sycl::plus<T>{},
                                              sycl::property::reduction::initialize_to_identity{});

      const auto workers = sycl::range<2>(num_elements_, param_dim_);

      cgh.parallel_for(workers, sum_reduction, [=](sycl::item<2> id, auto& sum_reduction) {
        const auto ItemRow = id.get_id(0) * output_dim_capture;
        const auto ItemCol = id.get_id(1);

        Eigen::Map<VectorT> residual_map(&residual_data_capture[ItemRow], output_dim_capture);
        Eigen::Map<VectorT> residual_plus_map(&residual_plus_data_capture[ItemRow + ItemCol * residuals_dim_capture],
                                              output_dim_capture);
        Eigen::Map<VectorT> jacobian_map(jacobian_data_capture, residuals_dim_capture, param_dim_capture);

        if (ItemRow < residuals_dim_capture && ItemCol < param_dim_capture) {
          models_sycl_plus[ItemCol].f(&input_capture[ItemRow], &observations_capture[ItemRow],
                                      residual_plus_map.data());

          if (ItemCol == 0) {
            model_sycl->f(&input_capture[ItemRow], &observations_capture[ItemRow], residual_map.data());
            sum_reduction += residual_map.squaredNorm();
          }
        }

        if (ItemRow < residuals_dim_capture && ItemCol < param_dim_capture) {
          jacobian_map.block(ItemRow, ItemCol, output_dim_capture, 1) = (residual_plus_map - residual_map) / g_step;
        }
      });
    });

    auto* JTJ_device = sycl::malloc_device<T>(param_dim_ * param_dim_, queue_);
    auto* JTb_device = sycl::malloc_device<T>(param_dim_, queue_);

    auto jtj_reset_event = queue_.memset(JTJ_device, 0, sizeof(T) * param_dim_ * param_dim_);
    auto jtb_reset_event = queue_.memset(JTb_device, 0, sizeof(T) * param_dim_);

    // // J := residuals x params
    // // A := J^T (params x residuals) (m x k)
    // // B := J (residuals x params) (k x n)

    // // m := params
    // // n := params
    // // k := residuals

    const oneapi::math::backend_selector<Backend> backend(queue_);

    // // C := J^T * J (params x params) (m x n)
    auto res_jtj = oneapi::math::blas::column_major::gemm(backend,
                                                          oneapi::math::transpose::trans,     // op(a)
                                                          oneapi::math::transpose::nontrans,  // op(b)
                                                          param_dim_,                         // m
                                                          param_dim_,                         // n
                                                          observation_dim_ * num_elements_,   // k
                                                          1.0,                                // alpha
                                                          jacobian_data_,                     // A*
                                                          observation_dim_ * num_elements_,   // lda
                                                          jacobian_data_,                     // B*
                                                          observation_dim_ * num_elements_,   // ldb
                                                          0.0,                                // beta
                                                          JTJ_device,                         // C*
                                                          param_dim_, {jac_event, jtj_reset_event});

    // // C := J^T * R (params x params) (m x n)
    auto res_jtb = oneapi::math::blas::column_major::gemv(backend,
                                                          oneapi::math::transpose::trans,    // op(A)
                                                          observation_dim_ * num_elements_,  // m
                                                          param_dim_,                        // n
                                                          1.0,                               // alpha
                                                          jacobian_data_,                    // A*
                                                          observation_dim_ * num_elements_,  // lda
                                                          residual_data_,                    // X*
                                                          1,                                 // incX
                                                          0.0,                               // beta
                                                          JTb_device,                        // Y
                                                          1,                                 // incY)
                                                          {jac_event, jtb_reset_event});

    sycl::event::wait({res_jtb, res_jtj});
    stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl kernel jacobian: took: {} us", stop);

    queue_.copy<T>(cost_reduction_, cost, 1).wait();

    t0.start();
    queue_.copy<T>(JTJ_device, JTJ, param_dim_ * param_dim_).wait();
    queue_.copy<T>(JTb_device, JTb, param_dim_).wait();
    stop = t0.stop();
    logger_->log(ILog::Level::DEBUG, "Sycl result data copy took: {} us", stop);

    sycl::free(models_sycl_plus, queue_);
    sycl::free(model_sycl, queue_);
    sycl::free(JTJ_device, queue_);
    sycl::free(JTb_device, queue_);
  }

 private:
  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;

  using VectorT = Eigen::Vector<T, Eigen::Dynamic>;
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  std::shared_ptr<ILog> logger_;

  const T* input_;
  const T* observations_;

  T* input_sycl_;
  T* observations_sycl_;
  T* cost_reduction_;
  T* jacobian_data_;
  T* residual_data_;
  T* residual_plus_data_;
  sycl::queue queue_;
};

}  // namespace moptim