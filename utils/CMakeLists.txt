add_library(utils
    ILog.cc
    ConsoleLogger.cc
    Timer.cc)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})