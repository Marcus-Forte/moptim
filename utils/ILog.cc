
#include "ILog.hh"

#include <chrono>
#include <ctime>
#include <format>

ILog::ILog() : level_(ILog::Level::DEBUG) {}
ILog::ILog(ILog::Level level) : level_(level) {}

ILog::~ILog() = default;

void ILog::setLevel(ILog::Level level) { level_ = level; }

std::string ILog::toString(ILog::Level level) {
  switch (level) {
    case ILog::Level::DEBUG:
      return "Debug";
    case ILog::Level::ERROR:
      return "Error";
    case ILog::Level::INFO:
      return "Info";
    case ILog::Level::WARNING:
      return "Warning";
    default:
      return "Unknown";
  }
}

std::string ILog::getTimeString() {
  std::time_t time;
  std::time(&time);
  const auto* timestamp = std::localtime(&time);
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto fraction = now - seconds;
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction).count();
  const std::string timestring =
      std::format("{}:{}:{}.{}", timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec, milliseconds);
  return timestring;
}