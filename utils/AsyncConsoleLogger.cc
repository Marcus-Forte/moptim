

#include "AsyncConsoleLogger.hh"

#include <iostream>

constexpr int LogQueueCapacity = 1024;
constexpr int ThreadPeriodMs = 100;

AsyncConsoleLogger::AsyncConsoleLogger() : log_queue_{LogQueueCapacity} {
  logger_thread_ = std::jthread([this](const std::stop_token& stoken) {
    while (!stoken.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ThreadPeriodMs));
      LogCommand command;
      while (!log_queue_.empty()) {
        if (log_queue_.pop(command)) {
          std::cout << command() << std::endl;
        }
      }
    }
  });
}

AsyncConsoleLogger::~AsyncConsoleLogger() {
  logger_thread_.request_stop();
  logger_thread_.join();
}

void AsyncConsoleLogger::log_impl(ILog::Level level, LogCommand&& message) const { log_queue_.push(message); }