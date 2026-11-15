#pragma once

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/Manifold.hh"
#include "moptim/Observer.hh"
#include "moptim/Result.hh"
#include "moptim/Status.hh"

namespace moptim {

/**
 * @brief Base of the iterative optimizers: owns the costs, the iteration loop and the workspace the
 * linear system of an iteration is assembled into. A derived class implements one iteration.
 *
 * The parameters are either a plain vector, built from its dimension, or live on an IManifold. `x`
 * has `ambientDim()` entries; the linear system and the step have `tangentDim()` (the same number
 * for a plain vector). The step is applied with the manifold's `plus`, by addition for a plain
 * vector. An optimizer and its costs are either all built from plain dimensions or all built from
 * the same manifold object.
 */
template <class T>
class IOptimizer {
 public:
  /// Euclidean parameters of `dimensions` entries.
  explicit IOptimizer(size_t dimensions) : IOptimizer(nullptr, dimensions, dimensions) {}

  /// Parameters on `manifold`, which must not be null.
  explicit IOptimizer(std::shared_ptr<const IManifold<T>> manifold)
      : IOptimizer(manifold, ambientDimOf(manifold), tangentDimOf(manifold)) {}
  virtual ~IOptimizer() = default;

  /// Runs a single iteration from `x` (`ambientDim()` entries), updating it in place.
  Status step(T* x) { return stepImpl(x, 0).status; }

  /// Iterates from `x` (`ambientDim()` entries), updating it in place, until an iteration ends the
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

  /// Adds a term to the objective. It must have been built for the same space as the optimizer:
  /// the same manifold object (both null for plain dimensions) and `paramDim() == ambientDim()`.
  void addCost(const std::shared_ptr<ICost<T>>& cost) {
    assert(cost && cost->manifold() == manifold_.get() && cost->paramDim() == ambient_dim_ &&
           cost->tangentDim() == tangent_dim_);
    costs_.push_back(cost);
  }
  void clearCosts() { costs_.clear(); }

  /// Entries of `x`.
  size_t ambientDim() const { return ambient_dim_; }
  /// Entries of a step: the size of the linear system.
  size_t tangentDim() const { return tangent_dim_; }
  /// The manifold of the parameters, null for plain dimensions.
  const IManifold<T>* manifold() const { return manifold_.get(); }

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
  size_t ambient_dim_;
  size_t tangent_dim_;
  std::shared_ptr<const IManifold<T>> manifold_;  ///< Null for plain dimensions.

  MatrixT hessian_;   ///< J^T J summed over the costs, tangent_dim_ square, filled by assembleLinearSystem().
  VectorT gradient_;  ///< J^T b summed over the costs, tangent_dim_ entries, filled by assembleLinearSystem().

  static size_t ambientDimOf(const std::shared_ptr<const IManifold<T>>& manifold) {
    assert(manifold);
    return manifold->ambientDim();
  }
  static size_t tangentDimOf(const std::shared_ptr<const IManifold<T>>& manifold) {
    assert(manifold);
    return manifold->tangentDim();
  }

  /// `x_plus = x [+] delta`: the manifold's plus, or a sum for plain dimensions. `x_plus` is
  /// `ambient_dim_` entries and does not alias `x` or `delta`.
  void applyStep(const T* x, const T* delta, T* x_plus) const {
    if (manifold_) {
      manifold_->plus(x, delta, x_plus);
    } else {
      Eigen::Map<VectorT>(x_plus, ambient_dim_) =
          Eigen::Map<const VectorT>(x, ambient_dim_) + Eigen::Map<const VectorT>(delta, ambient_dim_);
    }
  }

 private:
  IOptimizer(std::shared_ptr<const IManifold<T>> manifold, size_t ambient_dim, size_t tangent_dim)
      : ambient_dim_(ambient_dim),
        tangent_dim_(tangent_dim),
        manifold_(std::move(manifold)),
        hessian_(tangent_dim, tangent_dim),
        gradient_(tangent_dim),
        cost_hessian_(tangent_dim, tangent_dim),
        cost_gradient_(tangent_dim) {}

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
