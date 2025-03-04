cmake_minimum_required(VERSION 3.20)
project(moptim)
set(CMAKE_CXX_STANDARD 20)

include(CTest)

option(USE_CLANG_TIDY "Enable clang-tidy" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(code-format)

# \todo Clang tidy does not work with ACPP
if(USE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(Eigen3 REQUIRED)

add_subdirectory(utils)

add_library(moptim 
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc)
target_compile_options(moptim PUBLIC --acpp-targets=generic)
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim utils Eigen3::Eigen)

## Testing
if(BUILD_TESTING)
add_subdirectory(test)
endif()