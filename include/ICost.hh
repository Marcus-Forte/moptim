#pragma once

#include <Eigen/Dense>

class ICost {
 public:
  using SolveRhs = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double>;
  using Summation = std::tuple<Eigen::VectorXd, double>;

  virtual Summation computeResidual(const Eigen::VectorXd& x) = 0;
  virtual Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) = 0;

  /**
   * @brief Most efficient API
   *
   * @param x
   * @return SolveRhs
   */
  virtual SolveRhs computeHessian(const Eigen::VectorXd& x) = 0;
};