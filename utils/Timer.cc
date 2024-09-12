#include "Timer.hh"

void Timer::start() { start_ = std::chrono::high_resolution_clock::now(); }

uint64_t Timer::stop() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_)
      .count();
}