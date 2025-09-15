#pragma once

#include "ILog.hh"
#include "IOptimizer.hh"
#include "ISolver.hh"

namespace moptim {
template <class T>
class LevenbergMarquardt : public IOptimizer<T> {
 public:
  LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ILog>& logger, const std::shared_ptr<ISolver<T>>& solver);
  LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ILog>& logger);

  Status step(T* x) const override;
  Status optimize(T* x) const override;

 private:
  mutable double lm_init_lambda_factor_;
  mutable double lm_lambda_;
  size_t lm_iterations_ = 3;
  std::shared_ptr<ISolver<T>> solver_;
  std::shared_ptr<ILog> logger_;
};
}  // namespace moptim