#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "SE3.h"

#include <cassert>
#include <cmath>




namespace moptim {

template <class T>
struct SE3PlusOperator {
  static void plus(const T* x, const T* delta, T* out, size_t dimensions) {
    assert(dimensions == 6);

    const Eigen::Map<const Eigen::Matrix<T, 6, 1>> x_vec(x);
    const Eigen::Map<const Eigen::Matrix<T, 6, 1>> delta_vec(delta);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> out_vec(out);

    out_vec =
        se3Log(se3Exp(delta_vec) * se3Exp(x_vec));
  }
};

}  // namespace moptim