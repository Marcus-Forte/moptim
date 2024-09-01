#include "NullLogger.hh"

void NullLogger::log_impl(ILog::Level level, const std::string& message) const {}