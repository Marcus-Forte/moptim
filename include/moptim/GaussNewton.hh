#pragma once

#include "moptim/IOptimizer.hh"
#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class GaussNewton : public IOptimizer<T> {
 public:
  GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit GaussNewton(size_t dimensions);

 protected:
  Status stepImpl(T* x, size_t iteration) const override;

 private:
  std::shared_ptr<ISolver<T>> solver_;
};

}  // namespace moptim
