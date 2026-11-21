#pragma once

#include <chrono>
#include <memory>
#include <optional>

#include "moptim/IOptimizer.hh"
#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class LevenbergMarquardt : public IOptimizer<T> {
 public:
  /// Euclidean parameters of `dimensions` entries. The solver solves systems of that size.
  LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit LevenbergMarquardt(size_t dimensions);

  /// Parameters on `manifold`. The solver solves systems of `manifold->tangentDim()` entries.
  LevenbergMarquardt(std::shared_ptr<const IManifold<T>> manifold, const std::shared_ptr<ISolver<T>>& solver);
  explicit LevenbergMarquardt(std::shared_ptr<const IManifold<T>> manifold);

  /**
   * @brief Evaluates each trial point with a full linear system instead of the cost alone (default: off).
   *
   * The trial's cost is the one of that system, which equals computeCost() bitwise for the
   * ChunkedCost-based costs (AnalyticalCost, NumericalCost*), so x, the status, the iteration count and
   * the final cost are the same bit for bit with and without it. When the trial is accepted the next
   * iteration uses that system instead of assembling a new one: an accepted iteration then costs one
   * launch (one linear system) instead of two (a cost, then a linear system), and a rejected trial costs
   * a linear system instead of a cost, which is dearer. It pays when most trials are accepted and the
   * launch overhead matters (small problems, few iterations); it loses when many are rejected, and the
   * last accepted iteration of a run builds a system that is never used.
   *
   * Observers: onLinearSystem() fires once for each system that is used, as before, and not for the
   * systems of rejected trials; an accepted trial's system is reported at the start of the iteration
   * that uses it, with the time its assembly took.
   *
   * Do not use it with an ICost whose computeLinearSystem() cost differs from its computeCost(): the
   * optimizer would then compare trial costs with another function than the one it reports.
   */
  void setSpeculativeLinearSystems(bool enabled) { speculative_ = enabled; }
  bool speculativeLinearSystems() const { return speculative_; }

 protected:
  using typename IOptimizer<T>::StepResult;

  StepResult stepImpl(T* x, size_t iteration) override;
  void onOptimizeStart() override;
  bool shouldEmitFinishedEvent(Status status) const override;

 private:
  using typename IOptimizer<T>::VectorT;

  /// Damped steps tried per iteration before giving the iteration up.
  static constexpr size_t kMaxTrials = 3;
  /// The first damping factor is this fraction of the largest diagonal entry of J^T J.
  static constexpr T kInitialLambdaFactor = static_cast<T>(1e-7);

  std::shared_ptr<ISolver<T>> solver_;

  bool speculative_ = false;
  using typename IOptimizer<T>::MatrixT;
  MatrixT spec_hessian_;  ///< Linear system of the trial point (speculative mode), swapped in on acceptance.
  VectorT spec_gradient_;
  bool spec_event_pending_ = false;  ///< An accepted trial's system not yet reported to the observer.
  T spec_event_cost_ = T{0};
  std::chrono::nanoseconds spec_event_elapsed_{};

  /// Damping factor, carried from one iteration to the next. Empty until the first linear system
  /// of a run sets its scale.
  std::optional<T> lambda_;

  /// True while hessian_/gradient_ still describe the current x: the previous iteration of this
  /// run rejected all of its trials and left x untouched, so the next one need not reassemble them.
  bool linear_system_is_current_ = false;
  T linear_system_cost_ = T{0};  ///< Total cost at the x the linear system was assembled at.
  VectorT diagonal_;             ///< Diagonal of J^T J, before damping (tangent_dim_ entries).
  VectorT delta_;                ///< Step of the current trial (tangent_dim_ entries).
  VectorT trial_x_;              ///< x [+] delta_ (ambient_dim_ entries).
};
}  // namespace moptim
