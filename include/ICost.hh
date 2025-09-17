#pragma once

#include <cstddef>

namespace moptim {

template <class T>
class ICost {
 public:
  ICost(const ICost&) = delete;
  virtual ~ICost() = default;
  ICost(size_t dimensions, size_t num_elements) : dimensions_(dimensions), num_elements_(num_elements) {}

  /**
   * @brief Compute the cost given parameters x
   *
   * @param x Parameters
   * @return T Cost value
   */
  virtual T computeCost(const T* x) = 0;

  /**
   * @brief Compute the linear system: JTJ, JTb and cost
   *
   * @param x Parameters
   * @param JTJ Hessian (J^T * J)
   * @param JTb Gradient (J^T * b)
   * @param cost Cost value
   */
  virtual void computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) = 0;

 protected:
  const size_t dimensions_;
  const size_t num_elements_;
};
}  // namespace moptim