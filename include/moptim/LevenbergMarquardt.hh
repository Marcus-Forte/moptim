#pragma once

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
