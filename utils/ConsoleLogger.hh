#pragma once

#include "ILog.hh"

class ConsoleLogger : public ILog {
 public:
  ConsoleLogger();
  ConsoleLogger(ILog::Level level);

 private:
  void log_impl(ILog::Level level, LogCommand&& message) const override;
};