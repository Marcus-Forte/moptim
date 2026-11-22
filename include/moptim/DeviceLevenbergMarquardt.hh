#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "moptim/ExecutionContext.hh"
#include "moptim/Result.hh"
#include "moptim/Status.hh"
#include "moptim/detail/KernelName.hh"

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
#endif

namespace moptim {

/**
 * @brief A parameter space for device code: `kAmbientDim` entries of `x`, steps of `kTangentDim`, and a
 * static, device-callable `plus(x, delta, x_plus)` (`x_plus` does not alias the inputs).
 */
template <class M, class T>
concept DeviceManifold = requires(const T* x, const T* delta, T* x_plus) {
  { M::kAmbientDim } -> std::convertible_to<size_t>;
  { M::kTangentDim } -> std::convertible_to<size_t>;
  M::plus(x, delta, x_plus);
};

/// Plain vectors of `N` entries as a DeviceManifold (the same arithmetic as EuclideanManifold).
template <class T, size_t N>
struct EuclideanSpace {
  static constexpr size_t kAmbientDim = N;
  static constexpr size_t kTangentDim = N;
  static void plus(const T* x, const T* delta, T* x_plus) {
    for (size_t i = 0; i < N; ++i) {
      x_plus[i] = x[i] + delta[i];
    }
  }
};

namespace detail {

/// Name of the kernel of a DeviceLevenbergMarquardt.
template <class T, class Manifold, class TuId, class... Policies>
struct DeviceLMKernelName;

inline double deviceSqrt(double v) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sqrt(v);
#else
  return std::sqrt(v);
#endif
}
inline float deviceSqrt(float v) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sqrt(v);
#else
  return std::sqrt(v);
#endif
}
template <class T>
T devicePow(T base, T exponent) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::pow(base, exponent);
#else
  return std::pow(base, exponent);
#endif
}

/// Solves (L L^T) x = -b for a symmetric positive definite `n` x `n` matrix `A` (column-major, lower
/// triangle read), as EigenSolver does; `L` is scratch of n*n. Returns false if A is not positive
/// definite or the result is not finite.
template <class T>
bool choleskySolve(const T* A, const T* b, T* x, T* L, size_t n) {
  for (size_t j = 0; j < n; ++j) {
    T diagonal = A[j + j * n];
    for (size_t k = 0; k < j; ++k) {
      diagonal -= L[j + k * n] * L[j + k * n];
    }
    if (!(diagonal > T{0})) {
      return false;
    }
    const T pivot = deviceSqrt(diagonal);
    L[j + j * n] = pivot;
    for (size_t i = j + 1; i < n; ++i) {
      T value = A[i + j * n];
      for (size_t k = 0; k < j; ++k) {
        value -= L[i + k * n] * L[j + k * n];
      }
      L[i + j * n] = value / pivot;
    }
  }
  // L y = -b, then L^T x = y.
  for (size_t i = 0; i < n; ++i) {
    T value = -b[i];
    for (size_t k = 0; k < i; ++k) {
      value -= L[i + k * n] * x[k];
    }
    x[i] = value / L[i + i * n];
  }
  for (size_t ii = n; ii-- > 0;) {
    T value = x[ii];
    for (size_t k = ii + 1; k < n; ++k) {
      value -= L[k + ii * n] * x[k];
    }
    x[ii] = value / L[ii + ii * n];
  }
  for (size_t i = 0; i < n; ++i) {
    if (!(x[i] - x[i] == T{0})) {  // NaN or infinite
      return false;
    }
  }
  return true;
}

}  // namespace detail

/**
 * @brief Levenberg-Marquardt that runs a whole optimize() in one kernel launch, for small problems.
 *
 * On a host context the same loop runs on the calling thread (no launch at all), so the class is the
 * small-problem optimizer of both backends.
 *
 * LevenbergMarquardt orchestrates an iteration on the host and launches the costs' kernels, about one
 * launch per iteration with speculative linear systems and a FusedCost. For a problem of a few hundred
 * to a few thousand elements in a few dimensions the launches are most of the time. This optimizer
 * runs the iterations themselves in a kernel, on one work-item: the linear systems (each cost's
 * partitions in order, the chunked work-item code, the costs summed in order), the damped Cholesky
 * solve, the step on the manifold, the trial point's linear system and the acceptance test. It follows
 * LevenbergMarquardt with speculative linear systems step for step (the same damping, trials, rho,
 * convergence tests and status), so it ends where LevenbergMarquardt does up to rounding: the device
 * pass computes some things in another order (scalar Eigen in models, its own Cholesky).
 *
 * It is opt-in and has requirements of its own:
 *  - The parameter space is a compile-time DeviceManifold (`Manifold`), with the same `plus` as the
 *    manifold the costs were built on, and at most 64 tangent dimensions.
 *  - `Costs` are ChunkedCost types (AnalyticalCost, NumericalCost*) sharing one ExecutionContext, whose
 *    models' `setState()` is device code too (it runs in the kernel).
 *  - There is no observer, and the work runs on one work-item: use it where launches, not arithmetic,
 *    dominate. LevenbergMarquardt stays the reference and the choice for large problems.
 *
 * The costs are shared: change their data or model between runs as usual (setData(), setModel()).
 */
template <class T, class Manifold, class... Costs>
class DeviceLevenbergMarquardt {
  static_assert(sizeof...(Costs) >= 1, "a DeviceLevenbergMarquardt needs at least one cost");
  static_assert(DeviceManifold<Manifold, T>, "Manifold must be a DeviceManifold");

 public:
  static constexpr size_t kAmbient = Manifold::kAmbientDim;
  static constexpr size_t kTangent = Manifold::kTangentDim;
  static_assert(kTangent >= 1 && kTangent <= 64, "DeviceLevenbergMarquardt handles 1 to 64 tangent dimensions");

  /// @throws std::invalid_argument for a null cost, costs on different contexts, or costs whose
  ///   dimensions are not the manifold's.
  explicit DeviceLevenbergMarquardt(std::shared_ptr<Costs>... costs) : costs_(std::move(costs)...) {
    context_ = std::get<0>(costs_)->context();
    std::apply(
        [this](const auto&... cost) {
          const bool valid = ((cost && cost->context() == context_ && cost->paramDim() == kAmbient &&
                               cost->tangentDim() == kTangent) &&
                              ...);
          if (!valid) {
            throw std::invalid_argument(
                "moptim: the costs of a DeviceLevenbergMarquardt must share their ExecutionContext and have the "
                "manifold's dimensions");
          }
        },
        costs_);
    workspace_ = static_cast<Workspace*>(context_->allocateShared(sizeof(Workspace), alignof(Workspace)));
  }

  ~DeviceLevenbergMarquardt() { context_->deallocate(workspace_); }

  DeviceLevenbergMarquardt(const DeviceLevenbergMarquardt&) = delete;
  DeviceLevenbergMarquardt& operator=(const DeviceLevenbergMarquardt&) = delete;

  void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }

  /// Iterates from `x` (kAmbient entries), updating it in place: one launch.
  Result<T> optimize(T* x) { return optimizeImpl(x, std::index_sequence_for<Costs...>{}); }

  const std::shared_ptr<ExecutionContext>& context() const { return context_; }

 private:
  static constexpr size_t kCount = sizeof...(Costs);
  static constexpr size_t kMaxTrials = 3;  // LevenbergMarquardt's

  /// Everything the kernel works in (USM).
  struct Workspace {
    T x[kAmbient];
    T trial_x[kAmbient];
    T hessian[kTangent * kTangent];
    T gradient[kTangent];
    T spec_hessian[kTangent * kTangent];
    T spec_gradient[kTangent];
    T damped[kTangent * kTangent];
    T factor[kTangent * kTangent];
    T diagonal[kTangent];
    T delta[kTangent];
    T cost_hessian[kCount][kTangent * kTangent];
    T cost_gradient[kCount][kTangent];
    T cost_value[kCount];
    Status status;
    size_t iterations;
    T final_cost;
  };

  template <size_t... Is>
  Result<T> optimizeImpl(T* x, std::index_sequence<Is...>) {
    const std::tuple<typename Costs::LoopRef...> refs{std::get<Is>(costs_)->prepareLoop()...};
    Workspace* ws = workspace_;
    for (size_t i = 0; i < kAmbient; ++i) {
      ws->x[i] = x[i];
    }
    const size_t max_iterations = max_iterations_;
    // One work-group: on the host backend the loop runs on the calling thread.
    detail::launchNamed<detail::DeviceLMKernelName<T, Manifold, detail::CurrentTuId, typename Costs::FusedPolicy...>>(
        *context_, 1, [refs, ws, max_iterations](size_t) { run(refs, ws, max_iterations, std::index_sequence<Is...>{}); });
    for (size_t i = 0; i < kAmbient; ++i) {
      x[i] = ws->x[i];
    }
    return Result<T>{ws->status, ws->iterations, ws->final_cost};
  }

  /// The linear system of every cost at `x`, summed in cost order into `hessian`/`gradient` (what
  /// IOptimizer::assembleLinearSystemInto() computes); returns the total cost.
  template <size_t... Is>
  static T assemble(const std::tuple<typename Costs::LoopRef...>& refs, Workspace* ws, const T* x, T* hessian,
                    T* gradient, std::index_sequence<Is...>) {
    (Costs::template loopEvaluate<Manifold, true>(std::get<Is>(refs), x, ws->cost_hessian[Is],
                                                  ws->cost_gradient[Is], &ws->cost_value[Is]),
     ...);
    for (size_t i = 0; i < kTangent * kTangent; ++i) {
      hessian[i] = T{0};
    }
    for (size_t i = 0; i < kTangent; ++i) {
      gradient[i] = T{0};
    }
    T total{0};
    for (size_t c = 0; c < kCount; ++c) {
      for (size_t i = 0; i < kTangent * kTangent; ++i) {
        hessian[i] += ws->cost_hessian[c][i];
      }
      for (size_t i = 0; i < kTangent; ++i) {
        gradient[i] += ws->cost_gradient[c][i];
      }
      total += ws->cost_value[c];
    }
    return total;
  }

  static bool isDeltaSmall(const T* delta) {
    T largest{0};
    for (size_t i = 0; i < kTangent; ++i) {
      const T magnitude = delta[i] < T{0} ? -delta[i] : delta[i];
      largest = magnitude > largest ? magnitude : largest;
    }
    if constexpr (std::is_same_v<T, float>) {
      return largest < 1e-5F;  // Convergence.hh's g_small_delta_f
    } else {
      return largest < detail::deviceSqrt(std::numeric_limits<T>::epsilon());
    }
  }

  static bool isCostSmall(T cost) {
    if constexpr (std::is_same_v<T, float>) {
      return cost < T(1e-10);
    } else {
      return cost < T(1e-80);
    }
  }

  /// LevenbergMarquardt::optimize() with speculative linear systems, on one work-item.
  template <size_t... Is>
  static void run(const std::tuple<typename Costs::LoopRef...>& refs, Workspace* ws, size_t max_iterations,
                  std::index_sequence<Is...> seq) {
    constexpr T kInitialLambdaFactor = static_cast<T>(1e-7);
    T lambda{0};
    bool have_lambda = false;
    bool system_is_current = false;
    T system_cost{0};
    Status status = Status::STEP_OK;
    size_t iterations = 0;
    T final_cost{0};

    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
      if (iteration == 0 || !system_is_current) {
        system_cost = assemble(refs, ws, ws->x, ws->hessian, ws->gradient, seq);
        for (size_t i = 0; i < kTangent; ++i) {
          ws->diagonal[i] = ws->hessian[i + i * kTangent];
        }
        system_is_current = true;
      }
      const T initial_cost = system_cost;
      if (!have_lambda) {
        T largest{0};
        for (size_t i = 0; i < kTangent; ++i) {
          const T magnitude = ws->diagonal[i] < T{0} ? -ws->diagonal[i] : ws->diagonal[i];
          largest = magnitude > largest ? magnitude : largest;
        }
        lambda = kInitialLambdaFactor * largest;
        have_lambda = true;
      }

      Status step_status = Status::STEP_OK;
      T step_cost = initial_cost;
      T nu = 2;
      for (size_t trial = 0; trial < kMaxTrials; ++trial) {
        for (size_t i = 0; i < kTangent * kTangent; ++i) {
          ws->damped[i] = ws->hessian[i];
        }
        for (size_t i = 0; i < kTangent; ++i) {
          const T scaled = lambda * ws->diagonal[i];
          ws->damped[i + i * kTangent] = ws->diagonal[i] + scaled;
        }
        if (!detail::choleskySolve(ws->damped, ws->gradient, ws->delta, ws->factor, kTangent)) {
          for (size_t i = 0; i < kTangent; ++i) {
            ws->delta[i] = T{0};
          }
          lambda *= nu;
          nu = 2 * nu;
          continue;
        }

        Manifold::plus(ws->x, ws->delta, ws->trial_x);
        const T trial_cost = assemble(refs, ws, ws->trial_x, ws->spec_hessian, ws->spec_gradient, seq);
        T predicted{0};
        for (size_t i = 0; i < kTangent; ++i) {
          const T scaled = lambda * ws->delta[i];
          predicted += ws->delta[i] * (scaled - ws->gradient[i]);
        }
        const T rho = (initial_cost - trial_cost) / predicted;

        if (rho < T{0} || rho != rho) {
          if (isDeltaSmall(ws->delta)) {
            step_status = isCostSmall(trial_cost) ? Status::CONVERGED : Status::SMALL_DELTA;
            step_cost = initial_cost;
            break;
          }
          lambda *= nu;
          nu = 2 * nu;
          continue;
        }

        for (size_t i = 0; i < kAmbient; ++i) {
          ws->x[i] = ws->trial_x[i];
        }
        for (size_t i = 0; i < kTangent * kTangent; ++i) {
          ws->hessian[i] = ws->spec_hessian[i];
        }
        for (size_t i = 0; i < kTangent; ++i) {
          ws->gradient[i] = ws->spec_gradient[i];
          ws->diagonal[i] = ws->hessian[i + i * kTangent];
        }
        system_cost = trial_cost;
        system_is_current = true;
        const T t = 2 * rho - 1;
        const T shrink = 1 - detail::devicePow(t, T{3});
        lambda *= shrink > T(1.0 / 3.0) ? shrink : T(1.0 / 3.0);
        step_cost = trial_cost;
        break;
      }

      iterations = iteration + 1;
      status = step_status;
      final_cost = step_cost;
      if (step_status != Status::STEP_OK) {
        break;
      }
    }
    if (status == Status::STEP_OK) {
      status = Status::MAX_ITERATIONS_REACHED;
    }
    if (max_iterations == 0) {
      // optimize() then evaluates the cost at x.
      final_cost = assemble(refs, ws, ws->x, ws->hessian, ws->gradient, seq);
    }
    ws->status = status;
    ws->iterations = iterations;
    ws->final_cost = final_cost;
  }

  std::tuple<std::shared_ptr<Costs>...> costs_;
  std::shared_ptr<ExecutionContext> context_;
  Workspace* workspace_ = nullptr;
  size_t max_iterations_ = 15;
};

}  // namespace moptim
