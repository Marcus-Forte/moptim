#pragma once

#include "ILog.hh"
#include "IOptmizer.hh"

class GaussNewton : public IOptimizer {
 public:
  GaussNewton();
  GaussNewton(const std::shared_ptr<ILog>& logger);
  double step(Eigen::VectorXd& x) const override;
  Status optimize(Eigen::VectorXd& x) const override;
};