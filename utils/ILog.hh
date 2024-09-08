#pragma once

#include <format>
#include <string>

class ILog {
 public:
  enum class Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };
  ILog();
  ILog(Level level);
  virtual ~ILog();

  void log(Level level, const std::string& message) const;
  void setLevel(Level level);

  virtual void log_impl(ILog::Level level, const std::string& message) const = 0;

 protected:
  static std::string toString(Level level);
  static std::string getTime();

  Level level_;
};