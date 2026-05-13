#pragma once

#include "ILog.hh"
#include "moptim/IOptimizer.h"
#include "moptim/ISolver.h"

namespace moptim {
template <class T>
class GaussNewton : public IOptimizer<T> {
 public:
  GaussNewton(size_t dimensions, const std::shared_ptr<ILog>& logger, const std::shared_ptr<ISolver<T>>& solver);
  GaussNewton(size_t dimensions, const std::shared_ptr<ILog>& logger);

  Status step(T* x) const override;
  Status optimize(T* x) const override;

 private:
  std::shared_ptr<ISolver<T>> solver_;
  std::shared_ptr<ILog> logger_;
};

}  // namespace moptim