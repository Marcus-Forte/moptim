#pragma once

#include <chrono>

class Timer {
 public:
  /**
   * @brief Start timer.
   *
   */
  void start();

  /**
   * @brief
   *
   * @return uint64_t Elapsed time in us.
   */
  uint64_t stop();

 private:
  std::chrono::high_resolution_clock::time_point start_;
};