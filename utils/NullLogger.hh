#pragma once

#include "ILog.hh"

class NullLogger : public ILog {
 public:
  NullLogger() : ILog(ILog::Level::ERROR) {}
  void log_impl(ILog::Level /*level*/, LogCommand&& /*message*/) const override {}
};