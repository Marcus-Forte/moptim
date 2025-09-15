#pragma once

#include <cstddef>

namespace moptim {
enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

template <class T>
class ICost {
 public:
  ICost(size_t dimensions, size_t num_elements) : dimensions_(dimensions), num_elements_(num_elements) {}
  virtual ~ICost() = default;

  virtual T computeCost(const T* x) = 0;

  /**
   * @brief Compute the linear system: JTJ, JTb and cost
   *
   * @param x
   * @param JTJ
   * @param JTb
   * @param cost
   */
  virtual void computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) = 0;

  /**
   * @brief Get number of elements to iterate over the cost
   *
   * @return size_t number of elements
   */
  size_t getNumElements() const { return num_elements_; }

 protected:
  const size_t dimensions_;
  const size_t num_elements_;
};
}  // namespace moptim