cmake_minimum_required(VERSION 3.20)
project(moptim)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

option(USE_CLANG_TIDY "Enable clang-tidy" OFF)
option(WITH_SYCL "Use SYCL compilation for parallel computation" OFF)
add_compile_definitions(EIGEN_ARM64_USE_SVE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(code-format)

# \todo Clang tidy does not work with sycl
if(USE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(utils)

add_library(moptim 
  src/NumericalCost.cc
  src/AnalyticalCost.cc
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc)

if(WITH_SYCL)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
  if(NOT COMPILER_SUPPORTS_FSYCL)
      message(FATAL_ERROR "SYCL compilation requires acpp compiler")
  endif()
  target_compile_options(moptim PUBLIC -fsycl -fsycl-targets=native_cpu)
  target_link_options(moptim PUBLIC -fsycl -fsycl-targets=native_cpu)
endif()
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim utils Eigen3::Eigen)

## Testing
if(BUILD_TESTING)
add_subdirectory(test)
endif()