#pragma once

#include <span>

#include "ISolver.hh"

template <class T>
class EigenSolver : public ISolver<T> {
 public:
  EigenSolver(const std::shared_ptr<ILog>& logger, size_t dimensions) : ISolver<T>(logger, dimensions) {}
  ~EigenSolver() override = default;

  void solve(std::span<const T> A, std::span<const T> b, std::span<T> x) const override;
};