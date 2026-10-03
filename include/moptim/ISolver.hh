#pragma once

#include <cstddef>

namespace moptim {

template <class T>
class ISolver {
 public:
  explicit ISolver(size_t dimensions) : dimensions_(dimensions) {}
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
  size_t dimensions_;
};
}  // namespace moptim
