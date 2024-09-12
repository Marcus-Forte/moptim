#pragma once

#include <format>
#include <string>

class ILog {
 public:
  enum class Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };
  ILog();
  ILog(Level level);
  virtual ~ILog();

  template <typename... Args>
  void log(Level level, std::format_string<Args...> fmt, Args&&... args) const {
    if (level < level_) {
      return;
    }

    auto&& time = getTimeString();
    auto&& levelstr = toString(level);
    auto&& fmt_msg = std::format(fmt, std::forward<Args>(args)...);
    const auto msg = std::format("[{}][{}]: {}", time, levelstr, fmt_msg);
    log_impl(level, msg);
  }
  void setLevel(Level level);

  virtual void log_impl(ILog::Level level, const std::string& message) const = 0;

 private:
  static std::string toString(Level level);
  static std::string getTimeString();

  Level level_;
};