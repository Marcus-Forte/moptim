#pragma once

#include "ISolver.hh"

template <class T>
class EigenSolver : public ISolver<T> {
 public:
  EigenSolver(const std::shared_ptr<ILog>& logger, size_t dimensions) : ISolver<T>(logger, dimensions) {}
  ~EigenSolver() override = default;

  void solve(const T* A, const T* b, T* x) const override;
};