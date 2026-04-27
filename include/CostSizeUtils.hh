#pragma once

#include <span>
#include <stdexcept>

namespace moptim::detail {
template <class T>
size_t inferNumElements(std::span<const T> input, std::span<const T> observations, size_t input_dim,
                        size_t observation_dim) {
  if (input_dim == 0 || observation_dim == 0) {
    throw std::invalid_argument("input_dim and observation_dim must be > 0");
  }
  if (input.size() % input_dim != 0) {
    throw std::invalid_argument("input size must be divisible by input_dim");
  }

  const size_t num_elements = input.size() / input_dim;
  if (observations.size() != num_elements * observation_dim) {
    throw std::invalid_argument("observations size must match num_elements * observation_dim");
  }
  return num_elements;
}
}  // namespace moptim::detail