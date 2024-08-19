#pragma once

#include <memory>
#include <vector>

#include "ICost.hh"

class IOptimize {
 public:
  virtual void optimize(double* x) const = 0;

  inline void addCost(const std::shared_ptr<ICost>& cost) { costs_.push_back(cost); }

  inline void clearCosts() { costs_.clear(); }

 private:
  std::vector<std::shared_ptr<ICost>> costs_;
};