
#include "ILog.hh"

#include <chrono>
#include <ctime>
#include <format>

ILog::ILog() : level_(ILog::Level::WARNING) {}
ILog::ILog(ILog::Level level) : level_(level) {}

ILog::~ILog() = default;

void ILog::setLevel(ILog::Level level) { level_ = level; }

std::string ILog::toString(ILog::Level level) {
  switch (level) {
    case ILog::Level::TRACE:
      return "Trace";
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
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm timestamp{};
  localtime_r(&time, &timestamp);
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto fraction = now - seconds;
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction).count();
  return std::format("{}:{}:{}.{}", timestamp.tm_hour, timestamp.tm_min, timestamp.tm_sec, milliseconds);
}