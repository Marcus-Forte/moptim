#pragma once

#include <Eigen/Dense>

class ICost {
 public:
  using SolveRhs = std::pair<Eigen::MatrixXd, Eigen::VectorXd>;

  virtual double computeError(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const = 0;

  virtual SolveRhs computeHessian(const Eigen::VectorXd& x) const = 0;

 protected:
};