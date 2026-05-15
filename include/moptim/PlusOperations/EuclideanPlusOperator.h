#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <thread>

namespace moptim {

template <class T>
struct EuclideanPlusOperator {
  static void plus(const T* x, const T* delta, T* out, size_t dimensions) {
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const Eigen::Map<const VectorT> x_vec(x, dimensions);
    const Eigen::Map<const VectorT> delta_vec(delta, dimensions);
    Eigen::Map<VectorT> out_vec(out, dimensions);

    out_vec = x_vec + delta_vec;

    // for( size_t i = 0; i < dimensions; ++i) {
    //   out[i] = x[i] +  delta[i];
    // }
  }
};

}  // namespace moptim