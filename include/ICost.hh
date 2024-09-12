#pragma once

#include <Eigen/Dense>

class ICost {
 public:
  // JTJ, JTb, cost
  using SolveRhs = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double>;

  virtual double computeCost(const Eigen::VectorXd& x) = 0;

  /**
   * @brief Most efficient API
   *
   * @param x
   * @return SolveRhs
   */
  virtual SolveRhs computeLinearSystem(const Eigen::VectorXd& x) = 0;
};