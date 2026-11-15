#pragma once

#include <memory>

#include "moptim/IOptimizer.hh"
#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class GaussNewton : public IOptimizer<T> {
 public:
  /// Euclidean parameters of `dimensions` entries. The solver solves systems of that size.
  GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver);
  explicit GaussNewton(size_t dimensions);

  /// Parameters on `manifold`. The solver solves systems of `manifold->tangentDim()` entries.
  GaussNewton(std::shared_ptr<const IManifold<T>> manifold, const std::shared_ptr<ISolver<T>>& solver);
  explicit GaussNewton(std::shared_ptr<const IManifold<T>> manifold);

 protected:
  using typename IOptimizer<T>::StepResult;

  StepResult stepImpl(T* x, size_t iteration) override;

 private:
  using typename IOptimizer<T>::VectorT;

  std::shared_ptr<ISolver<T>> solver_;
  VectorT delta_;   ///< Step of the current iteration, tangent_dim_ entries.
  VectorT x_plus_;  ///< x [+] delta_: the manifold's plus does not work in place. Empty without a manifold.
};

}  // namespace moptim
