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

# clang-tidy cannot parse a -fsycl compilation, so the option is kept for the day it can.
if(USE_CLANG_TIDY)
  message(WARNING "USE_CLANG_TIDY does not work with SYCL (moptim is always built with -fsycl); expect clang-tidy errors")
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
  set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(Eigen3 REQUIRED)

# SYCL is a hard requirement: the library and everything that links it is compiled with a DPC++
# (Intel oneAPI / intel/llvm) clang++. Comma separated -fsycl-targets list, provided by the configure
# preset (see CMakePresets.json), e.g. native_cpu,nvptx64-nvidia-cuda,spir64.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" MOPTIM_COMPILER_SUPPORTS_FSYCL)
if(NOT MOPTIM_COMPILER_SUPPORTS_FSYCL)
  message(FATAL_ERROR "moptim requires a SYCL compiler (DPC++ clang++ with -fsycl), but ${CMAKE_CXX_COMPILER} does not "
                      "support it. Use e.g. -DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ or 'cmake --preset default'.")
endif()
set(SYCL_TARGETS "native_cpu" CACHE STRING "Comma separated list passed to -fsycl-targets")
# Empty by default. Value of -mcpu for the native_cpu device code generation (x86 hosts need e.g.
# x86-64-v3 to get AVX in device code); it only takes effect at link time, where that code is generated.
set(MOPTIM_NATIVE_CPU_ARCH "" CACHE STRING "-mcpu value for the native_cpu device code")

add_library(moptim
  src/EigenSolver.cc
  src/ExecutionContext.cc
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc
  src/NormalEquations.cc)

string(REPLACE "," ";" MOPTIM_SYCL_TARGET_LIST "${SYCL_TARGETS}")
set(MOPTIM_SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})

# Eigen 3.4's NEON code crashes the native_cpu DEVICE front end (arm_neon.h name mangling), so in the
# device pass of a -fsycl compilation Eigen runs scalar, while the host pass keeps its SIMD code.
# The device-only define is given to the compiler (not set in a header), so it applies to every
# translation unit that links moptim whatever its include order, and to every target of SYCL_TARGETS:
# `-Xsycl-target-frontend=<triple>` passes an option to the device front end of that target only.
# -ffp-contract=fast (GCC's default; clang's is `on`) lets the device pass contract a*b+c across
# statements, which measured 4 to 19% faster than `on` on the analytical costs (MOPTIM_FP_CONTRACT).
# Fixed-size Eigen members are shared between the host and the device pass (a model is copied into
# USM), so their layout must be identical: EIGEN_MAX_STATIC_ALIGN_BYTES=16 on both passes.
# SHELL: keeps CMake from de-duplicating the repeated -Xsycl-target-frontend halves.
set(MOPTIM_FP_CONTRACT "fast" CACHE STRING "Value of -ffp-contract for moptim and its consumers (fast, on or off)")
set(MOPTIM_DEVICE_ONLY_FLAGS "")
foreach(triple IN LISTS MOPTIM_SYCL_TARGET_LIST)
  list(APPEND MOPTIM_DEVICE_ONLY_FLAGS "SHELL:-Xsycl-target-frontend=${triple} -DEIGEN_DONT_VECTORIZE")
endforeach()
target_compile_options(moptim PUBLIC ${MOPTIM_SYCL_FLAGS} -ffp-contract=${MOPTIM_FP_CONTRACT} ${MOPTIM_DEVICE_ONLY_FLAGS})
target_compile_definitions(moptim PUBLIC EIGEN_MAX_STATIC_ALIGN_BYTES=16)
target_link_options(moptim PUBLIC ${MOPTIM_SYCL_FLAGS} -ffp-contract=${MOPTIM_FP_CONTRACT})
if(MOPTIM_NATIVE_CPU_ARCH)
  target_link_options(moptim PUBLIC "SHELL:-Xsycl-target-frontend=native_cpu -mcpu=${MOPTIM_NATIVE_CPU_ARCH}")
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