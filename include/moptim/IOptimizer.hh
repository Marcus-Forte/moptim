#pragma once

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/Observer.hh"
#include "moptim/Result.hh"
#include "moptim/Status.hh"

namespace moptim {

/**
 * @brief Base of the iterative optimizers: owns the costs, the iteration loop and the workspace the
 * linear system of an iteration is assembled into. A derived class implements one iteration.
 */
template <class T>
class IOptimizer {
 public:
  explicit IOptimizer(size_t dimensions)
      : dimensions_(dimensions),
        hessian_(dimensions, dimensions),
        gradient_(dimensions),
        cost_hessian_(dimensions, dimensions),
        cost_gradient_(dimensions) {}
  virtual ~IOptimizer() = default;

  /// Runs a single iteration from `x` (`dimensions` entries), updating it in place.
  Status step(T* x) { return stepImpl(x, 0).status; }

  /// Iterates from `x` (`dimensions` entries), updating it in place, until an iteration ends the
  /// run or the iteration budget is spent.
  Result<T> optimize(T* x);

  /**
   * @brief Attach a non-owning telemetry observer. Passing nullptr detaches.
   *
   * The observer receives structured events; it is responsible for any
   * formatting, logging or recording. Without an observer no event payload is
   * gathered, so the optimizer stays free of logging dependencies.
   */
  void setObserver(IOptimizerObserver<T>* observer) { observer_ = observer; }

  void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }

  /// Adds a term to the objective. Its parameter dimension must match the optimizer's.
  void addCost(const std::shared_ptr<ICost<T>>& cost) {
    assert(cost && cost->paramDim() == dimensions_);
    costs_.push_back(cost);
  }
  void clearCosts() { costs_.clear(); }

 protected:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using Clock = std::chrono::steady_clock;

  /// Outcome of one iteration.
  struct StepResult {
    Status status;
    /// Total cost at the `x` the iteration returned, when the iteration already evaluated it. It
    /// spares optimize() a pass over every cost to report the final cost.
    std::optional<T> cost;
  };

  /// Performs a single optimization step at the given iteration index, used internally by
  /// optimize()'s loop. `step()` calls this with iteration 0 for manual single-stepping.
  virtual StepResult stepImpl(T* x, size_t iteration) = 0;

  /// Hook invoked once at the start of optimize(), before any iterations, letting subclasses
  /// reset per-run state (e.g. Levenberg-Marquardt's damping factor).
  virtual void onOptimizeStart() {}

  /// Whether optimize() should emit a FINISHED IterationEvent after its loop ends. Default: always.
  /// Overridden by LevenbergMarquardt, whose stepImpl() already emits FINISHED when it
  /// self-terminates (CONVERGED/SMALL_DELTA), so only the MAX_ITERATIONS_REACHED case remains.
  virtual bool shouldEmitFinishedEvent(Status status) const {
    (void)status;
    return true;
  }

  /// Sums the linear system of every cost at `x` into hessian_ and gradient_, reports it to the
  /// observer and returns the total cost.
  T assembleLinearSystem(const T* x, size_t iteration);

  /// Total cost at `x`.
  T evaluateCost(const T* x);

  /// Time stamp for event payloads. The clock is only read when an observer is attached.
  Clock::time_point now() const { return observer_ ? Clock::now() : Clock::time_point{}; }

  std::vector<std::shared_ptr<ICost<T>>> costs_;
  IOptimizerObserver<T>* observer_ = nullptr;
  size_t max_iterations_ = 15;
  size_t dimensions_;

  MatrixT hessian_;   ///< J^T J summed over the costs, filled by assembleLinearSystem().
  VectorT gradient_;  ///< J^T b summed over the costs, filled by assembleLinearSystem().

 private:
  MatrixT cost_hessian_;  ///< Linear system of a single cost, before it is added to the sum.
  VectorT cost_gradient_;
};

template <class T>
Result<T> IOptimizer<T>::optimize(T* x) {
  onOptimizeStart();

  Result<T> result;
  std::optional<T> cost;
  const auto start = now();

  for (size_t i = 0; i < max_iterations_; ++i) {
    const StepResult step = stepImpl(x, i);
    result.iterations = i + 1;
    result.status = step.status;
    cost = step.cost;

    if (step.status != Status::STEP_OK) {
      break;
    }
  }

  if (result.status == Status::STEP_OK) {
    result.status = Status::MAX_ITERATIONS_REACHED;
  }

  result.final_cost = cost ? *cost : evaluateCost(x);

  if (observer_ && shouldEmitFinishedEvent(result.status)) {
    IterationEvent<T> event;
    event.iteration = result.iterations;
    event.phase = Phase::FINISHED;
    event.status = result.status;
    event.cost = result.final_cost;
    event.elapsed = Clock::now() - start;
    observer_->onIteration(event);
  }

  return result;
}

template <class T>
T IOptimizer<T>::assembleLinearSystem(const T* x, size_t iteration) {
  const auto start = now();

  hessian_.setZero();
  gradient_.setZero();
  T total_cost{0};
  for (const auto& cost : costs_) {
    T cost_value{0};
    cost->computeLinearSystem(x, cost_hessian_.data(), cost_gradient_.data(), cost_value);
    hessian_ += cost_hessian_;
    gradient_ += cost_gradient_;
    total_cost += cost_value;
  }

  if (observer_) {
    LinearSystemEvent<T> event;
    event.iteration = iteration;
    event.cost = total_cost;
    event.elapsed = Clock::now() - start;
    observer_->onLinearSystem(event);
  }

  return total_cost;
}

template <class T>
T IOptimizer<T>::evaluateCost(const T* x) {
  T total_cost{0};
  for (const auto& cost : costs_) {
    total_cost += cost->computeCost(x);
  }
  return total_cost;
}

}  // namespace moptim
