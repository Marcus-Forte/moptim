#pragma once

#include <memory>
#include <vector>

#include "moptim/ICost.h"
#include "moptim/Status.h"

namespace moptim::constants {}

namespace moptim {

template <class T>
class IOptimizer {
 public:
  IOptimizer(size_t dimensions) : dimensions_(dimensions) {}
  virtual ~IOptimizer() = default;

  virtual Status step(T* x) const = 0;
  virtual Status optimize(T* x) const = 0;

  void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }

  void addCost(const std::shared_ptr<ICost<T>>& cost) { costs_.push_back(cost); }
  void clearCosts() { costs_.clear(); }

 protected:
  std::vector<std::shared_ptr<ICost<T>>> costs_;
  size_t max_iterations_ = 15;
  size_t dimensions_;
};
}  // namespace moptim