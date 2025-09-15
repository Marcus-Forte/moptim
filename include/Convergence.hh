#pragma once

#include <Eigen/Dense>

namespace moptim {

constexpr double g_small_cost_d = 1e-80;
constexpr double g_small_cost_f = 1e-10;

static inline bool isDeltaSmall(const double* vec, size_t dimensions) {
  Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>> vec_map(vec, dimensions);
  const auto epsilon = vec_map.array().abs().maxCoeff();
  return epsilon < sqrt(std::numeric_limits<double>::epsilon());
}

static inline bool isDeltaSmall(const float* vec, size_t dimensions) {
  Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>> vec_map(vec, dimensions);
  const auto epsilon = vec_map.array().abs().maxCoeff();
  return epsilon < 1e-5;
}

// Stub implementations
inline bool isCostSmall(float cost) { return cost < g_small_cost_f; }
inline bool isCostSmall(double cost) { return cost < g_small_cost_d; }
}  // namespace moptim