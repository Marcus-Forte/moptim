#pragma once

#include "ILog.hh"

class ConsoleLogger : public ILog {
 private:
  void log_impl(ILog::Level level, LogCommand&& message) const override;
};