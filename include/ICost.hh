#pragma once

#include <Eigen/Dense>

class ICost {
 public:
  using SolveRhs = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double>;

  virtual double getCost(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const = 0;

  virtual SolveRhs computeHessian(const Eigen::VectorXd& x) const = 0;

 protected:
};