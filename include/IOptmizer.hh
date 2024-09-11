#pragma once

#include <memory>
#include <vector>

#include "ICost.hh"
#include "ILog.hh"

class IOptimizer {
 public:
  IOptimizer() = default;
  IOptimizer(const std::shared_ptr<ILog>& logger) : logger_(logger) {}
  enum class Status { CONVERGED = 0, SMALL_DELTA = 1, MAX_ITERATIONS_REACHED = 2 };
  virtual double step(Eigen::VectorXd& x) const = 0;
  virtual Status optimize(Eigen::VectorXd& x) const = 0;

  inline void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }
  inline void addCost(const std::shared_ptr<ICost>& cost) { costs_.push_back(cost); }
  inline void clearCosts() { costs_.clear(); }

 protected:
  static inline bool isSmall(const Eigen::VectorXd& vec) {
    const auto epsilon = vec.array().abs().maxCoeff();
    return epsilon < sqrt(std::numeric_limits<double>::epsilon());
  }

  std::shared_ptr<ILog> logger_;
  std::vector<std::shared_ptr<ICost>> costs_;
  size_t max_iterations_ = 10;
};