#pragma once

#include <cassert>
#include <mdspan>
#include <numeric>
#include <ranges>
#include <span>

#include "IModel.hh"

namespace moptim::detail {

template <class T>
T computeCost(std::span<const T> x, size_t param_dim, IModel<T>& model,
              std::mdspan<const T, std::dextents<size_t, 2>> input_elements,
              std::mdspan<const T, std::dextents<size_t, 2>> observation_elements, std::span<T> residual_out) {
  assert(x.size() == param_dim);
  model.setup(x);
  const auto* input_data = input_elements.data_handle();
  const auto* observation_data = observation_elements.data_handle();
  const size_t input_dim = input_elements.extent(1);
  const size_t observation_dim = observation_elements.extent(1);
  for (const size_t i : std::views::iota(size_t{0}, input_elements.extent(0))) {
    const auto row = i * observation_dim;
    model.f(std::span<const T>(input_data + (i * input_dim), input_dim),
            std::span<const T>(observation_data + row, observation_dim), residual_out.subspan(row, observation_dim));
  }
  return std::transform_reduce(residual_out.begin(), residual_out.end(), residual_out.begin(), T{0});
}

}  // namespace moptim::detail
