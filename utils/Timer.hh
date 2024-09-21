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
   * @brief
   * @return uint64_t Elapsed time in us since start().
   */
  uint64_t stop();

 private:
  std::chrono::high_resolution_clock::time_point start_;
};