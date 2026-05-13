#pragma once

#include <memory>

#include "ILog.hh"

template <class T>
class ISolver {
 public:
  ISolver(const std::shared_ptr<ILog>& logger, size_t dimensions) : logger_(logger), dimensions_(dimensions) {}
  virtual ~ISolver() = default;

  /**
   * @brief Solve the linear system `Ax = b` for x.
   *
   * @param[in] A Matrix A
   * @param[in] b Vector b
   * @param[out] x
   */
  virtual void solve(const T* A, const T* b, T* x) const = 0;

 protected:
  std::shared_ptr<ILog> logger_;
  size_t dimensions_;
};