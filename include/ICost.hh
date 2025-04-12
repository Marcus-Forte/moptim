#pragma once

#include <Eigen/Dense>

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

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

 protected:
  inline static SolveRhs Reduction(SolveRhs a, const SolveRhs& b) {
    auto& [JTJ_a, JTb_a, residual_a] = a;
    const auto& [JTJ_b, JTb_b, residual_b] = b;
    JTJ_a += JTJ_b;
    JTb_a += JTb_b;
    residual_a += residual_b;

    return a;
  }
};