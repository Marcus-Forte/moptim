#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <limits>

namespace moptim {

// A cost below this is treated as an exact fit.
constexpr double g_small_cost_d = 1e-80;
constexpr double g_small_cost_f = 1e-10;

// A step whose largest entry is below this no longer moves the parameters.
constexpr double g_small_delta_f = 1e-5;

inline bool isDeltaSmall(const double* data, size_t size) {
  const Eigen::Map<const Eigen::VectorXd> delta(data, size);
  return delta.lpNorm<Eigen::Infinity>() < std::sqrt(std::numeric_limits<double>::epsilon());
}

inline bool isDeltaSmall(const float* data, size_t size) {
  const Eigen::Map<const Eigen::VectorXf> delta(data, size);
  return delta.lpNorm<Eigen::Infinity>() < g_small_delta_f;
}

inline bool isCostSmall(float cost) { return cost < g_small_cost_f; }
inline bool isCostSmall(double cost) { return cost < g_small_cost_d; }

}  // namespace moptim
