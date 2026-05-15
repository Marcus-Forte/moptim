add_library(utils
    ILog.cc
    AsyncConsoleLogger.cc
    ConsoleLogger.cc
    Timer.cc)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(utils PRIVATE -fexperimental-library) # for jthread
endif()
