#pragma once

#include <format>
#include <functional>
#include <string>

class ILog {
 public:
  using LogCommand = std::function<std::string()>;  // \todo fix scope?
  enum class Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };
  ILog();
  ILog(Level level);
  virtual ~ILog();

  template <typename... Args>
  void log(Level level, std::format_string<Args...> fmt, Args&&... args) const {
    if (level < level_) {
      return;
    }

    // Move the formatting as a command to the implementation to call at its own time.
    auto formatter = [=]() mutable -> std::string {
      auto time = getTimeString();
      auto levelstr = toString(level);
      auto fmt_msg = std::format(fmt, std::forward<Args>(args)...);
      return std::format("[{}][{}]: {}", time, levelstr, fmt_msg);
    };

    log_impl(level, std::move(formatter));
  }
  void setLevel(Level level);

  virtual void log_impl(ILog::Level level, LogCommand&& command) const = 0;

 private:
  static std::string toString(Level level);
  static std::string getTimeString();

  Level level_;
};