#pragma once

#include <cassert>
#include <mdspan>
#include <numeric>
#include <span>

namespace moptim::detail {

template <class Model, class T>
T computeCost(std::span<const T> x, size_t param_dim, Model& model,
              std::mdspan<const T, std::dextents<size_t, 2>> inputs,
              std::mdspan<const T, std::dextents<size_t, 2>> observations, std::span<T> residual_out) {
  assert(x.size() == param_dim);
  const size_t num_elements = inputs.extent(0);
  const size_t obs_dim = observations.extent(1);
  std::mdspan residuals_md(residual_out.data(), num_elements, obs_dim);
  model.residuals(x, inputs, observations, residuals_md);
  return std::transform_reduce(residual_out.begin(), residual_out.end(), T{0}, std::plus<T>{},
                               [](T r) { return r * r; });
}

}  // namespace moptim::detail
