#pragma once

#include <Eigen/Dense>
#include <cassert>

#include "moptim/PlusOperations/SE3.h"

namespace moptim {

/**
 * @brief Plus operator for a 15-DOF state: SE(3) ⊕ R^9.
 *
 * State layout: [position(3), rotation(3), velocity(3), gyro_bias(3),
 * accel_bias(3)]
 *
 * The first 6 components (position + rotation) use SE(3) exponential map
 * composition. The remaining 9 components (velocity + biases) use standard
 * Euclidean addition.
 *
 * This satisfies the IPlus concept: `plus(x, delta, out, dimensions)`.
 */
template <class T> struct SE3xEuclideanPlusOperator {
  static void plus(const T *x, const T *delta, T *out, size_t dimensions) {
    assert(dimensions == 15);

    // SE(3) composition for pose (first 6 DOF)
    const Eigen::Map<const Eigen::Matrix<T, 6, 1>> x_se3(x);
    const Eigen::Map<const Eigen::Matrix<T, 6, 1>> delta_se3(delta);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> out_se3(out);

    out_se3 = se3Log(se3Exp(delta_se3) * se3Exp(x_se3));

    // Euclidean addition for velocity and biases (last 9 DOF)
    for (size_t i = 6; i < 15; ++i) {
      out[i] = x[i] + delta[i];
    }
  }
};

} // namespace moptim
