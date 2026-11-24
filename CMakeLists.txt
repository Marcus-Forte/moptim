cmake_minimum_required(VERSION 3.20)
project(moptim)
include(CTest)

# A standalone configure without a build type would compile without optimization, and unoptimized
# Eigen code is far slower. A parent project keeps control of its own build type.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
option(USE_CLANG_TIDY "Enable clang-tidy" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(code-format)

add_compile_options("-Wall")

if(USE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(Eigen3 REQUIRED)

# oneTBB runs the partitions of large costs on several threads (src/ExecutionContext.cc only: no public
# header names it). Without it, or with the option OFF, every launch runs on the calling thread.
option(MOPTIM_WITH_TBB "Run the partitions of large costs on oneTBB" ON)
if(MOPTIM_WITH_TBB)
  # A parent project (mslam) may have found TBB already: its imported target is visible here.
  if(NOT TARGET TBB::tbb)
    find_package(TBB)
  endif()
  if(NOT TARGET TBB::tbb)
    message(WARNING "moptim: oneTBB not found, building without it: costs are evaluated sequentially "
                    "(configure with -DMOPTIM_WITH_TBB=OFF to silence this)")
  endif()
endif()
message(STATUS "moptim: oneTBB ${MOPTIM_WITH_TBB}")

add_library(moptim
  src/EigenSolver.cc
  src/ExecutionContext.cc
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc
  src/NormalEquations.cc)

if(MOPTIM_WITH_TBB AND TARGET TBB::tbb)
  target_link_libraries(moptim PRIVATE TBB::tbb)
  target_compile_definitions(moptim PRIVATE MOPTIM_HAS_TBB=1)
endif()

# Core lib.
target_include_directories(moptim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The public headers use concepts, so consumers need at least C++20.
target_compile_features(moptim PUBLIC cxx_std_20)
target_link_libraries(moptim PUBLIC Eigen3::Eigen)
add_library(moptim::moptim ALIAS moptim)

## Testing
# Build tests only for a standalone moptim build. As a subproject the parent
# project drives CTest, and moptim is added EXCLUDE_FROM_ALL, so its test
# binaries are never built and would surface as dangling `<target>_NOT_BUILT`
# CTest failures.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
add_subdirectory(test)
endif()