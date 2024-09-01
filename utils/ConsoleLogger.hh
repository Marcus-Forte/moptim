#pragma once

#include "ILog.hh"

class ConsoleLogger : public ILog {
 public:
  void log_impl(ILog::Level level, const std::string& message) const override;
};