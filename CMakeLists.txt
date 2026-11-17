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
option(WITH_SYCL "Use SYCL compilation for parallel computation" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(code-format)

add_compile_options("-Wall")

# \todo Clang tidy does not work with sycl
if(USE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(Eigen3 REQUIRED)

add_library(moptim
  src/EigenSolver.cc
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc
  src/NormalEquations.cc)

if(WITH_SYCL)
  # check this link to select via env variable: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
  add_compile_definitions(EIGEN_ARM64_USE_SVE)
  add_compile_definitions(EIGEN_DONT_VECTORIZE)
  find_package(oneMath REQUIRED)
  message(STATUS "Found oneMath: ${oneMath_DIR}")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
  if(NOT COMPILER_SUPPORTS_FSYCL)
      message(FATAL_ERROR "SYCL compilation requires dpcpp compiler")
  endif()
  # Comma separated -fsycl-targets list, provided by the configure preset (see CMakePresets.json),
  # e.g: native_cpu,nvptx64-nvidia-cuda,spir64 (specific GPU triplet is also possible, e.g: nvidia_gpu_sm_86).
  set(SYCL_TARGETS "" CACHE STRING "Comma separated list passed to -fsycl-targets")
  if(NOT SYCL_TARGETS)
      message(FATAL_ERROR "WITH_SYCL=ON requires SYCL_TARGETS, e.g. use 'cmake --preset sycl' or -DSYCL_TARGETS=native_cpu")
  endif()
  set(SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})
  target_compile_options(moptim PUBLIC ${SYCL_FLAGS})
  target_link_options(moptim PUBLIC ${SYCL_FLAGS})
  # The onemath library must have been compiled in the base container.
  link_libraries(ONEMATH::onemath ONEMATH::onemath_blas_netlib gfortran) # Netlib
  # link_libraries(ONEMATH::onemath ONEMATH::onemath_blas_cublas cuda) # Cuda + Cublas
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