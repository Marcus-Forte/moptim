#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <limits>
#include <memory>
#include <new>
#include <oneapi/math.hpp>
#include <span>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/IModel.hh"

namespace moptim {

namespace detail {
/// Frees USM memory on the queue it was allocated with.
struct UsmDeleter {
  std::optional<sycl::queue> queue;  // optional: a default-constructed deleter must not create a queue
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr && queue) {
      sycl::free(ptr, *queue);
    }
  }
};

template <class U>
using UsmPtr = std::unique_ptr<U, UsmDeleter>;

/// Allocates `count` elements of device USM; throws std::bad_alloc on failure.
template <class U>
UsmPtr<U> makeDeviceArray(sycl::queue& queue, size_t count) {
  U* ptr = sycl::malloc_device<U>(count > 0 ? count : 1, queue);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return UsmPtr<U>(ptr, UsmDeleter{queue});
}
}  // namespace detail

/**
 * @brief Forward-difference numerical cost evaluated on a SYCL device.
 *
 * Euclidean parameters only. Same semantics as NumericalCostForwardEuler: inputs hold
 * `input_dim` values per element, observations `observation_dim`, only the leading
 * `active_param_dim` parameters are differentiated and the remaining rows/columns of JTJ and
 * entries of JTb are zero. `input` and `observations` must outlive the cost when the device can
 * access host memory (CPU devices); otherwise they are copied to the device at construction.
 *
 * The model is copied byte-wise to the device, so it must be trivially copyable, and
 * `residual()` runs on a work-item private copy of it.
 */
template <class T, class Model, oneapi::math::backend Backend = oneapi::math::backend::netlib>
  requires NumericalModel<Model, T>
class NumericalCostSycl : public ICost<T> {
 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const sycl::queue& queue, std::span<const T> input, std::span<const T> observations,
                    size_t input_dim, size_t observation_dim, size_t param_dim, size_t num_elements,
                    size_t active_param_dim = 0, Model model = Model{})
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements, active_param_dim),
        queue_(queue),
        model_(std::move(model)) {
    if constexpr (Backend == oneapi::math::backend::netlib) {
      if (!queue_.get_device().is_cpu()) {
        throw std::runtime_error("NumericalCostSycl: the netlib oneMath backend only supports CPU devices");
      }
    }
    if (active_param_dim_ > param_dim_) {
      throw std::invalid_argument("NumericalCostSycl: active_param_dim exceeds param_dim");
    }
    if (input.size() < input_dim_ * num_elements_ || observations.size() < observation_dim_ * num_elements_) {
      throw std::invalid_argument("NumericalCostSycl: input/observations spans are smaller than dim * num_elements");
    }

    const size_t input_count = input_dim_ * num_elements_;
    const size_t observation_count = observation_dim_ * num_elements_;

    if (queue_.get_device().has(sycl::aspect::usm_system_allocations)) {
      // Host memory is directly accessible from the device: no copy needed.
      input_ptr_ = input.data();
      observations_ptr_ = observations.data();
    } else {
      input_device_ = detail::makeDeviceArray<T>(queue_, input_count);
      observations_device_ = detail::makeDeviceArray<T>(queue_, observation_count);
      queue_.copy<T>(input.data(), input_device_.get(), input_count);
      queue_.copy<T>(observations.data(), observations_device_.get(), observation_count);
      queue_.wait_and_throw();
      input_ptr_ = input_device_.get();
      observations_ptr_ = observations_device_.get();
    }

    cost_reduction_ = detail::makeDeviceArray<T>(queue_, 1);
    // J: (observation_dim * num_elements) x active_param_dim, column major.
    jacobian_ = detail::makeDeviceArray<T>(queue_, observation_count * active_param_dim_);
    residual_ = detail::makeDeviceArray<T>(queue_, observation_count);
  }

  /// \todo if this is called before jacobian, we don't have to compute cost again.
  T computeCost(const T* x) override {
    std::vector<Model> host_model(1, model_);
    host_model[0].setState(x);
    auto model_device = detail::makeDeviceArray<Model>(queue_, 1);
    auto x_device = detail::makeDeviceArray<T>(queue_, param_dim_);
    queue_.copy<Model>(host_model.data(), model_device.get(), 1);
    queue_.copy<T>(x, x_device.get(), param_dim_);
    queue_.wait_and_throw();  // host buffers above stay alive until here

    computeResiduals(model_device.get(), x_device.get());
    queue_.wait_and_throw();

    T result;
    queue_.copy<T>(cost_reduction_.get(), &result, 1).wait();
    return result;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    const size_t active = active_param_dim_;
    const size_t residual_count = observation_dim_ * num_elements_;
    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());

    // Evaluation points: index 0 is x, index 1 + i is x with the i-th parameter perturbed.
    std::vector<Model> host_models(1 + active, model_);
    std::vector<T> host_points(param_dim_ * (1 + active));
    for (size_t i = 0; i < 1 + active; ++i) {
      T* point = host_points.data() + i * param_dim_;
      std::copy(x, x + param_dim_, point);
      if (i > 0) {
        point[i - 1] += g_step;
      }
      host_models[i].setState(point);
    }

    auto models_device = detail::makeDeviceArray<Model>(queue_, 1 + active);
    auto points_device = detail::makeDeviceArray<T>(queue_, host_points.size());
    queue_.copy<Model>(host_models.data(), models_device.get(), 1 + active);
    queue_.copy<T>(host_points.data(), points_device.get(), host_points.size());
    queue_.wait_and_throw();  // host buffers above stay alive until here

    const auto compute_residuals_event = computeResiduals(models_device.get(), points_device.get());

    // One work-item per (element, active parameter): forward difference written straight to J.
    const auto jac_event = queue_.submit([&](sycl::handler& cgh) {
      cgh.depends_on(compute_residuals_event);
      const T* input = input_ptr_;
      const T* observations = observations_ptr_;
      const T* residual = residual_.get();
      T* jacobian = jacobian_.get();
      const Model* models = models_device.get();
      const T* points = points_device.get();
      const size_t input_dim = input_dim_;
      const size_t observation_dim = observation_dim_;
      const size_t param_dim = param_dim_;
      const T inv_step = T{1} / g_step;

      cgh.parallel_for(sycl::range<2>(num_elements_, active), [=](sycl::item<2> id) {
        const size_t element = id.get_id(0);
        const size_t col = id.get_id(1);
        T* out = jacobian + col * residual_count + element * observation_dim;
        Model local_model = models[1 + col];
        local_model.residual(points + (1 + col) * param_dim, input + element * input_dim,
                             observations + element * observation_dim, out);
        for (size_t k = 0; k < observation_dim; ++k) {
          out[k] = (out[k] - residual[element * observation_dim + k]) * inv_step;
        }
      });
    });

    // Zero blocks for the inactive parameters; the BLAS calls only write the leading block.
    auto JTJ_device = detail::makeDeviceArray<T>(queue_, param_dim_ * param_dim_);
    auto JTb_device = detail::makeDeviceArray<T>(queue_, param_dim_);
    const auto fill_jtj = queue_.fill<T>(JTJ_device.get(), T{0}, param_dim_ * param_dim_);
    const auto fill_jtb = queue_.fill<T>(JTb_device.get(), T{0}, param_dim_);

    const oneapi::math::backend_selector<Backend> backend(queue_);

    // J := residuals x active, A := J^T, C := J^T * J (upper triangle only)
    const auto res_jtj = oneapi::math::blas::column_major::syrk(
        backend, oneapi::math::uplo::upper, oneapi::math::transpose::trans,
        static_cast<std::int64_t>(active),                // n
        static_cast<std::int64_t>(residual_count),        // k
        T{1}, jacobian_.get(),                            // alpha, A*
        static_cast<std::int64_t>(residual_count),        // lda
        T{0}, JTJ_device.get(),                           // beta, C*
        static_cast<std::int64_t>(param_dim_),            // ldc
        {jac_event, fill_jtj});

    // Copy the upper part to the lower part
    const auto mirror_jtj = queue_.submit([&](sycl::handler& cgh) {
      cgh.depends_on(res_jtj);
      T* jtj = JTJ_device.get();
      const size_t param_dim = param_dim_;
      cgh.parallel_for(sycl::range<2>(active, active), [=](sycl::item<2> id) {
        const size_t row = id.get_id(0);
        const size_t col = id.get_id(1);
        if (row > col) {
          jtj[row + param_dim * col] = jtj[col + param_dim * row];
        }
      });
    });

    const auto res_jtb = oneapi::math::blas::column_major::gemv(
        backend, oneapi::math::transpose::trans,
        static_cast<std::int64_t>(residual_count),  // m
        static_cast<std::int64_t>(active),          // n
        T{1}, jacobian_.get(),                      // alpha, A*
        static_cast<std::int64_t>(residual_count),  // lda
        residual_.get(), 1,                         // X*, incX
        T{0}, JTb_device.get(), 1,                  // beta, Y*, incY
        {jac_event, fill_jtb});

    sycl::event::wait_and_throw({res_jtb, mirror_jtj});

    queue_.copy<T>(cost_reduction_.get(), &cost, 1).wait();
    queue_.copy<T>(JTJ_device.get(), JTJ, param_dim_ * param_dim_).wait();
    queue_.copy<T>(JTb_device.get(), JTb, param_dim_).wait();
  }

 private:
  /// Residuals at `x_device` (evaluation point 0 of `models`) and their squared norm in cost_reduction_.
  sycl::event computeResiduals(const Model* models, const T* x_device) {
    return queue_.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction<T>(cost_reduction_.get(), T{0}, sycl::plus<T>{},
                                    sycl::property::reduction::initialize_to_identity{});

      const T* input = input_ptr_;
      const T* observations = observations_ptr_;
      T* residual = residual_.get();
      const size_t input_dim = input_dim_;
      const size_t observation_dim = observation_dim_;

      cgh.parallel_for(sycl::range<1>(num_elements_), sum, [=](sycl::item<1> id, auto& reduction) {
        const size_t element = id.get_id();
        T* out = residual + element * observation_dim;
        Model local_model = models[0];
        local_model.residual(x_device, input + element * input_dim, observations + element * observation_dim, out);
        T squared_norm{0};
        for (size_t k = 0; k < observation_dim; ++k) {
          squared_norm += out[k] * out[k];
        }
        reduction += squared_norm;
      });
    });
  }

  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;
  using ICost<T>::active_param_dim_;

  sycl::queue queue_;
  Model model_;  ///< Prototype every evaluation point is copied from.

  // Either borrowed host memory (host accessible devices) or the device copies below.
  const T* input_ptr_{nullptr};
  const T* observations_ptr_{nullptr};
  detail::UsmPtr<T> input_device_;
  detail::UsmPtr<T> observations_device_;

  detail::UsmPtr<T> cost_reduction_;
  detail::UsmPtr<T> jacobian_;
  detail::UsmPtr<T> residual_;
};

}  // namespace moptim
