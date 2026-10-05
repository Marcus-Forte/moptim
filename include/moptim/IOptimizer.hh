#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/Observer.hh"
#include "moptim/Result.hh"
#include "moptim/Status.hh"

namespace moptim::constants {}

namespace moptim {

template <class T>
class IOptimizer {
 public:
  IOptimizer(size_t dimensions) : dimensions_(dimensions) {}
  virtual ~IOptimizer() = default;

  virtual Status step(T* x) const { return stepImpl(x, 0); }
  virtual Result<T> optimize(T* x) const;

  /**
   * @brief Attach a non-owning telemetry observer. Passing nullptr detaches.
   *
   * The observer receives structured events; it is responsible for any
   * formatting, logging or recording. Without an observer no event payload is
   * gathered, so the optimizer stays free of logging dependencies.
   */
  void setObserver(IOptimizerObserver<T>* observer) { observer_ = observer; }

  void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }

  void addCost(const std::shared_ptr<ICost<T>>& cost) { costs_.push_back(cost); }
  void clearCosts() { costs_.clear(); }

 protected:
  /// Performs a single optimization step at the given iteration index, used internally by
  /// optimize()'s loop. `step()` calls this with iteration 0 for manual single-stepping.
  virtual Status stepImpl(T* x, size_t iteration) const = 0;

  /// Hook invoked once at the start of optimize(), before any iterations, letting subclasses
  /// reset per-run state (e.g. Levenberg-Marquardt's damping factor).
  virtual void onOptimizeStart() const {}

  /// Whether optimize() should emit a FINISHED IterationEvent after its loop ends. Default: always.
  /// Overridden by LevenbergMarquardt, whose stepImpl() already emits FINISHED when it
  /// self-terminates (CONVERGED/SMALL_DELTA), so only the MAX_ITERATIONS_REACHED case remains.
  virtual bool shouldEmitFinishedEvent(Status status) const {
    (void)status;
    return true;
  }

  std::vector<std::shared_ptr<ICost<T>>> costs_;
  IOptimizerObserver<T>* observer_ = nullptr;
  size_t max_iterations_ = 15;
  size_t dimensions_;
};

template <class T>
Result<T> IOptimizer<T>::optimize(T* x) const {
  onOptimizeStart();

  Result<T> result;
  const auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < max_iterations_; ++i) {
    const auto status = stepImpl(x, i);
    result.iterations = i + 1;
    result.status = status;

    if (status != Status::STEP_OK) {
      break;
    }
  }

  if (result.status == Status::STEP_OK) {
    result.status = Status::MAX_ITERATIONS_REACHED;
  }

  result.final_cost = T{};
  for (const auto& cost : costs_) {
    result.final_cost += cost->computeCost(x);
  }

  if (observer_ && shouldEmitFinishedEvent(result.status)) {
    IterationEvent<T> event;
    event.iteration = result.iterations;
    event.phase = Phase::FINISHED;
    event.status = result.status;
    event.cost = result.final_cost;
    event.elapsed = std::chrono::steady_clock::now() - start;
    observer_->onIteration(event);
  }

  return result;
}

}  // namespace moptim
