#pragma once

#include <cstddef>

namespace moptim {

template <class PlusOperator, class T>
concept IPlus = requires(const T* x, const T* delta, T* out,
                                      size_t dimensions) {
  { PlusOperator::plus(x, delta, out, dimensions) };
};

}  // namespace moptim