#pragma once

#include <memory>
#include <vector>

#include "ICost.hh"

class IOptimizer {
 public:
  virtual void step(Eigen::VectorXd& x) const = 0;
  virtual void optimize(Eigen::VectorXd& x) const = 0;

  inline void addCost(const std::shared_ptr<ICost>& cost) { costs_.push_back(cost); }

  inline void clearCosts() { costs_.clear(); }

 protected:
  std::vector<std::shared_ptr<ICost>> costs_;
};