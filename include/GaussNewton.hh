#pragma once

#include "IOptmizer.hh"

class GaussNewton : public IOptimizer {
 public:
  GaussNewton();
  void step(Eigen::VectorXd& x) const override;
  void optimize(Eigen::VectorXd& x) const override;

 private:
};