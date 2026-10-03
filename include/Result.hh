#pragma once

#include <cstddef>

#include "Status.hh"

namespace moptim {

/**
 * @brief Outcome of IOptimizer::optimize().
 *
 * Returned unconditionally and with no logging machinery involved, so
 * benchmarks and consumer libraries can consume the result programmatically.
 */
template <class T>
struct Result {
  Status status = Status::STEP_OK;
  size_t iterations = 0;
  T final_cost = T{};
};

}  // namespace moptim
