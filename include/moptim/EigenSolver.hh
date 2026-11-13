#pragma once

#include <Eigen/Cholesky>

#include "moptim/ISolver.hh"

namespace moptim {

/// Dense Cholesky (LL^T) solver. The factorization storage is allocated once and reused by every
/// solve().
template <class T>
class EigenSolver : public ISolver<T> {
 public:
  explicit EigenSolver(size_t dimensions);

  bool solve(const T* A, const T* b, T* x) override;

 private:
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> factorization_;
};
}  // namespace moptim
