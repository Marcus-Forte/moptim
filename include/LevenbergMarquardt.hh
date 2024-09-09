#pragma once

#include "IOptmizer.hh"

class LevenbergMarquardt : public IOptimizer {
 public:
  LevenbergMarquardt();
  LevenbergMarquardt(const std::shared_ptr<ILog>& logger);
  double step(Eigen::VectorXd& x) const override;
  Status optimize(Eigen::VectorXd& x) const override;

 private:
  mutable double lm_init_lambda_factor_;
  mutable double lm_lambda_;
  size_t lm_iterations_ = 3;
};