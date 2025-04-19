#pragma once

#include <chrono>

class Timer {
 public:
  Timer();
  /**
   * @brief (re)set timer.
   */
  void start();

  /**
   * @brief Elapsed time in microseconds or nanoseconds.
   *
   * @param inNanoseconds select nanoseconds or microseconds
   * @return uint64_t
   */
  uint64_t stop(bool inNanoseconds = false);

 private:
  std::chrono::high_resolution_clock::time_point start_;
};