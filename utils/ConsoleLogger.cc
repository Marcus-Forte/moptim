

#include "ConsoleLogger.hh"

#include <iostream>

void ConsoleLogger::log_impl(ILog::Level level, const std::string& message) const { std::cout << message << std::endl; }