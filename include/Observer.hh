#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "Status.hh"

namespace moptim {

/**
 * @brief Stage of an optimization that produced an event.
 */
enum class Phase : uint8_t {
  LINEAR_SYSTEM = 0,  ///< A J^T J / J^T r linear system was assembled.
  TRIAL = 1,          ///< A trial step was evaluated (Levenberg-Marquardt inner loop).
  ACCEPTED = 2,       ///< A trial step was accepted.
  REJECTED = 3,       ///< A trial step was rejected.
  FINISHED = 4        ///< The optimizer stopped.
};

/**
 * @brief Per-iteration telemetry emitted by an optimizer.
 *
 * This is plain data: no formatting, allocation, logging or I/O. Consumers
 * decide whether to record it, print it, or ignore it entirely. It is
 * dependency-free on purpose so the optimizer can be used without a logging
 * framework.
 */
template <class T>
struct IterationEvent {
  size_t iteration = 0;
  size_t trial = 0;
  Phase phase = Phase::TRIAL;
  Status status = Status::STEP_OK;
  T cost = T{};
  T previous_cost = T{};
  T rho = T{};
  T lambda = T{};
  T delta_norm = T{};
  std::chrono::nanoseconds elapsed{};
};

/**
 * @brief Telemetry for the (dominant) linear-system assembly of an iteration.
 */
template <class T>
struct LinearSystemEvent {
  size_t iteration = 0;
  T cost = T{};
  std::chrono::nanoseconds elapsed{};
};

/**
 * @brief Observation sink for optimizers.
 *
 * Attach with IOptimizer::setObserver(). The pointer is non-owning; the caller
 * must keep the observer alive for the duration of optimize()/step(). When no
 * observer is attached the optimizer performs no telemetry work at all.
 *
 * Presentation (console, file, network) is intentionally not part of this
 * interface and is not shipped with moptim. A consumer with its own logging
 * framework writes a tiny adapter that turns events into log records.
 */
template <class T>
class IOptimizerObserver {
 public:
  virtual ~IOptimizerObserver() = default;
  virtual void onIteration(const IterationEvent<T>& /*event*/) {}
  virtual void onLinearSystem(const LinearSystemEvent<T>& /*event*/) {}
};

}  // namespace moptim
