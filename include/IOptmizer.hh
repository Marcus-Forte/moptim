#pragma once

#include <memory>
#include <vector>

#include "ICost.hh"

class IOptimizer {
 public:
  enum class Status { CONVERGED, SMALL_DELTA, SMALL_COST, MAX_ITERATIONS_REACHED };
  virtual void step(Eigen::VectorXd& x) const = 0;
  virtual Status optimize(Eigen::VectorXd& x) const = 0;

  inline void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }
  inline void addCost(const std::shared_ptr<ICost>& cost) { costs_.push_back(cost); }
  inline void clearCosts() { costs_.clear(); }

  static inline bool isSmall(const Eigen::VectorXd& vec) {
    const auto epsilon = vec.array().abs().maxCoeff();
    return epsilon < sqrt(std::numeric_limits<double>::epsilon());
  }

 protected:
  std::vector<std::shared_ptr<ICost>> costs_;
  size_t max_iterations_ = 10;
};