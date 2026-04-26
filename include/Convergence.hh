#pragma once

#include <Eigen/Dense>
#include <span>

namespace moptim {

constexpr double g_small_cost_d = 1e-80;
constexpr double g_small_cost_f = 1e-10;

static inline bool isDeltaSmall(std::span<const double> vec) {
  Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>> vec_map(vec.data(), vec.size());
  const auto epsilon = vec_map.array().abs().maxCoeff();
  return epsilon < sqrt(std::numeric_limits<double>::epsilon());
}

static inline bool isDeltaSmall(std::span<const float> vec) {
  Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>> vec_map(vec.data(), vec.size());
  const auto epsilon = vec_map.array().abs().maxCoeff();
  return epsilon < 1e-5;
}

// Stub implementations
inline bool isCostSmall(float cost) { return cost < g_small_cost_f; }
inline bool isCostSmall(double cost) { return cost < g_small_cost_d; }
}  // namespace moptim