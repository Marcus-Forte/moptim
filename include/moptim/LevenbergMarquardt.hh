#pragma once

#include <memory>
#include <optional>

#include "moptim/IOptimizer.hh"
#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class LevenbergMarquardt : public IOptimizer<T> {
 public:
  LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit LevenbergMarquardt(size_t dimensions);

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
  VectorT diagonal_;             ///< Diagonal of J^T J, before damping.
  VectorT delta_;                ///< Step of the current trial.
  VectorT trial_x_;              ///< x + delta_.
};
}  // namespace moptim
