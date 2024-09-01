#pragma once

#include <Eigen/Dense>

class ICost {
 public:
  struct SolveRhs {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd residuals;
  };
  virtual double computeError(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const = 0;

 protected:
};