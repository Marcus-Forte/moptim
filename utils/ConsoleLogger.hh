#pragma once

#include "ILog.hh"

class ConsoleLogger : public ILog {
 private:
  void log_impl(ILog::Level level, const std::string& message) const override;
};