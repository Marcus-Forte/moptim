#pragma once

#include "IOptimizer.hh"
#include "ISolver.hh"

namespace moptim {
template <class T>
class GaussNewton : public IOptimizer<T> {
 public:
  GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit GaussNewton(size_t dimensions);

  Status step(T* x) const override;
  Result<T> optimize(T* x) const override;

 private:
  Status stepImpl(T* x, size_t iteration) const;

  std::shared_ptr<ISolver<T>> solver_;
};

}  // namespace moptim
