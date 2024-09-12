#pragma once

#include <memory>
#include <vector>

#include "ICost.hh"
#include "ILog.hh"

class IOptimizer {
 public:
  IOptimizer() = default;
  IOptimizer(const std::shared_ptr<ILog>& logger) : logger_(logger) {}

  enum class Status { STEP_OK = 0, CONVERGED = 1, SMALL_DELTA = 2, MAX_ITERATIONS_REACHED = 3, NUMERIC_ERROR = 4 };

  virtual Status step(Eigen::VectorXd& x) const = 0;
  virtual Status optimize(Eigen::VectorXd& x) const = 0;

  inline void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }
  inline void addCost(const std::shared_ptr<ICost>& cost) { costs_.push_back(cost); }
  inline void clearCosts() { costs_.clear(); }

 protected:
  static inline bool isDeltaSmall(const Eigen::VectorXd& vec) {
    const auto epsilon = vec.array().abs().maxCoeff();
    return epsilon < sqrt(std::numeric_limits<double>::epsilon());
  }

  std::shared_ptr<ILog> logger_;
  std::vector<std::shared_ptr<ICost>> costs_;
  size_t max_iterations_ = 15;
};