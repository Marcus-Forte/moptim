#include "Timer.hh"

Timer::Timer() { start(); }

void Timer::start() { start_ = std::chrono::high_resolution_clock::now(); }

uint64_t Timer::stop(bool inNanoseconds) {
  if (inNanoseconds) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_)
        .count();
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_)
      .count();
}