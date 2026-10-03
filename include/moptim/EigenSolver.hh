#pragma once

#include "moptim/ISolver.hh"

namespace moptim {
template <class T>
class EigenSolver : public ISolver<T> {
 public:
  explicit EigenSolver(size_t dimensions) : ISolver<T>(dimensions) {}
  ~EigenSolver() override = default;

  void solve(const T* A, const T* b, T* x) const override;
};
}  // namespace moptim
