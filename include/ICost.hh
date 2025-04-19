#pragma once

#include <Eigen/Dense>

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

class ICost {
 public:
  ICost(size_t num_elements) : num_elements_(num_elements) {}
  virtual ~ICost() = default;

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

  /**
   * @brief Get number of elements to iterate over the cost
   *
   * @return size_t number of elements
   */
  size_t getNumElements() const { return num_elements_; }

 protected:
  const size_t num_elements_;
};