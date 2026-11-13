#pragma once

#include <cstddef>

namespace moptim {

template <class T>
class ISolver {
 public:
  explicit ISolver(size_t dimensions) : dimensions_(dimensions) {}
  virtual ~ISolver() = default;

  /**
   * @brief Solve the linear system `A * x = -b` for x: the step that minimizes the quadratic model
   * with Hessian `A` and gradient `b`.
   *
   * @param[in] A Symmetric matrix, `dimensions x dimensions` in column-major order. Only its lower
   *   triangle is read.
   * @param[in] b Vector b, `dimensions` entries.
   * @param[out] x Solution, `dimensions` entries. Left in an unspecified state on failure.
   * @return false if the system could not be solved, e.g. because `A` is not positive definite.
   */
  [[nodiscard]] virtual bool solve(const T* A, const T* b, T* x) = 0;

 protected:
  size_t dimensions_;
};
}  // namespace moptim
