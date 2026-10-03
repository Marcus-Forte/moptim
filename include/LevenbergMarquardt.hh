#pragma once

#include "IOptimizer.hh"
#include "ISolver.hh"

namespace moptim {
template <class T>
class LevenbergMarquardt : public IOptimizer<T> {
 public:
  LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit LevenbergMarquardt(size_t dimensions);

  Status step(T* x) const override;
  Result<T> optimize(T* x) const override;

 private:
  Status stepImpl(T* x, size_t iteration) const;

  mutable T lm_init_lambda_factor_ = static_cast<T>(1e-7);
  mutable T lm_lambda_ = static_cast<T>(-1);
  size_t lm_iterations_ = 3;
  std::shared_ptr<ISolver<T>> solver_;
};
}  // namespace moptim
