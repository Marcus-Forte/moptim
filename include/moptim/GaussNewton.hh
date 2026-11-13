#pragma once

#include <memory>

#include "moptim/IOptimizer.hh"
#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class GaussNewton : public IOptimizer<T> {
 public:
  GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit GaussNewton(size_t dimensions);

 protected:
  using typename IOptimizer<T>::StepResult;

  StepResult stepImpl(T* x, size_t iteration) override;

 private:
  using typename IOptimizer<T>::VectorT;

  std::shared_ptr<ISolver<T>> solver_;
  VectorT delta_;  ///< Step of the current iteration.
};

}  // namespace moptim
