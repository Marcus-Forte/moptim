#pragma once

#include <boost/lockfree/spsc_queue.hpp>
#include <thread>

#include "ILog.hh"

class AsyncConsoleLogger : public ILog {
 public:
  AsyncConsoleLogger();
  ~AsyncConsoleLogger() override;

 private:
  void log_impl(ILog::Level level, LogCommand&& message) const override;
  std::jthread logger_thread_;
  mutable boost::lockfree::spsc_queue<LogCommand> log_queue_;
};