#pragma once

#include <Eigen/Dense>

namespace moptim {
template <class T>
static inline bool isDeltaSmall(const T* vec, size_t dimensions) {
  Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> vec_map(vec, dimensions);
  const auto epsilon = vec_map.array().abs().maxCoeff();
  return epsilon < sqrt(std::numeric_limits<T>::epsilon());
}
}  // namespace moptim