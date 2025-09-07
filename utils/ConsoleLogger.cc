

#include "ConsoleLogger.hh"

#include <iostream>

void ConsoleLogger::log_impl(ILog::Level level, LogCommand&& message) const { std::cout << message() << std::endl; }