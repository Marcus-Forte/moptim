#pragma once

#include <memory>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/Observer.hh"
#include "moptim/Result.hh"
#include "moptim/Status.hh"

namespace moptim::constants {}

namespace moptim {

template <class T>
class IOptimizer {
 public:
  IOptimizer(size_t dimensions) : dimensions_(dimensions) {}
  virtual ~IOptimizer() = default;

  virtual Status step(T* x) const = 0;
  virtual Result<T> optimize(T* x) const = 0;

  /**
   * @brief Attach a non-owning telemetry observer. Passing nullptr detaches.
   *
   * The observer receives structured events; it is responsible for any
   * formatting, logging or recording. Without an observer no event payload is
   * gathered, so the optimizer stays free of logging dependencies.
   */
  void setObserver(IOptimizerObserver<T>* observer) { observer_ = observer; }

  void setMaxIterations(size_t max_iterations) { max_iterations_ = max_iterations; }

  void addCost(const std::shared_ptr<ICost<T>>& cost) { costs_.push_back(cost); }
  void clearCosts() { costs_.clear(); }

 protected:
  std::vector<std::shared_ptr<ICost<T>>> costs_;
  IOptimizerObserver<T>* observer_ = nullptr;
  size_t max_iterations_ = 15;
  size_t dimensions_;
};
}  // namespace moptim
