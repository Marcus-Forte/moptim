

#include "ConsoleLogger.hh"

#include <format>
#include <iostream>

void ConsoleLogger::log_impl(ILog::Level level, const std::string& message) const {
  std::cout << std::format("[{}][{}]: {}", getTime(), toString(level), message) << std::endl;
}