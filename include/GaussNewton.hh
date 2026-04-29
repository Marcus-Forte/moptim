#pragma once

#include "ILog.hh"
#include "IOptimizer.hh"
#include "ISolver.hh"

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