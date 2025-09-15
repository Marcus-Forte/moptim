

#include "ConsoleLogger.hh"

#include <iostream>

ConsoleLogger::ConsoleLogger() : ILog(ILog::Level::DEBUG) {}
ConsoleLogger::ConsoleLogger(ILog::Level level) : ILog(level) {}
void ConsoleLogger::log_impl(ILog::Level level, LogCommand&& message) const { std::cout << message() << std::endl; }