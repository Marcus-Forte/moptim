#pragma once

#include <cstdint>

namespace moptim {
/**
 * @brief Optimizer status codes.
 *
 */
enum class Status : uint8_t {
  STEP_OK = 0,
  CONVERGED = 1,
  SMALL_DELTA = 2,
  MAX_ITERATIONS_REACHED = 3,
  NUMERIC_ERROR = 4
};
}  // namespace moptim