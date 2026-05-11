#pragma once

#include <format>
#include <functional>
#include <source_location>
#include <string>

class ILog {
 public:
  using LogCommand = std::function<std::string()>;
  enum class Level { TRACE = 0, DEBUG = 1, INFO = 2, WARNING = 3, ERROR = 4 };

  template <typename... Args>
  struct FormatWithLocation {
    std::format_string<Args...> fmt;
    std::source_location loc;
    const char* file;

    static consteval const char* extractFilename(const char* path) {
      const char* name = path;
      while (*path) {
        if (*path == '/') name = path + 1;
        ++path;
      }
      return name;
    }

    template <typename T>
    consteval FormatWithLocation(T&& f, std::source_location l = std::source_location::current())
        : fmt(std::forward<T>(f)), loc(l), file(extractFilename(l.file_name())) {}
  };

  ILog();
  ILog(Level level);
  virtual ~ILog();

  template <typename... Args>
  void log(Level level, FormatWithLocation<std::type_identity_t<Args>...> fmt, Args&&... args) const {
    if (level < level_) {
      return;
    }

    auto file = fmt.file;
    auto line = fmt.loc.line();
    auto formatter = [=, fmt_str = fmt.fmt]() mutable -> std::string {
      auto time = getTimeString();
      auto levelstr = toString(level);
      auto fmt_msg = std::format(fmt_str, std::forward<Args>(args)...);
      return std::format("[{}][{}][{}:{}]: {}", time, levelstr, file, line, fmt_msg);
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