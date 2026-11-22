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

# SYCL is opt-in. Without it moptim is plain C++ (any C++23 compiler, e.g. GCC) and runs everything on the
# host backend. With MOPTIM_WITH_SYCL=ON the library and everything that links it is compiled with a DPC++
# (Intel oneAPI / intel/llvm) clang++, and a context can run on either backend (ExecutionContext::Backend,
# MOPTIM_BACKEND=host|sycl for the default one). WITH_SYCL is the option's former name.
option(MOPTIM_WITH_SYCL "Build the SYCL backend (needs a DPC++ clang++)" ${WITH_SYCL})
if(MOPTIM_WITH_SYCL)
  # Comma separated -fsycl-targets list, provided by the configure preset (see CMakePresets.json), e.g.
  # native_cpu,nvptx64-nvidia-cuda,spir64.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" MOPTIM_COMPILER_SUPPORTS_FSYCL)
  if(NOT MOPTIM_COMPILER_SUPPORTS_FSYCL)
    message(FATAL_ERROR "MOPTIM_WITH_SYCL needs a SYCL compiler (DPC++ clang++ with -fsycl), but ${CMAKE_CXX_COMPILER} "
                        "does not support it. Use e.g. -DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ or 'cmake --preset sycl'.")
  endif()
  set(SYCL_TARGETS "native_cpu" CACHE STRING "Comma separated list passed to -fsycl-targets")
  # Empty by default. Value of -mcpu for the native_cpu device code generation (x86 hosts need e.g.
  # x86-64-v3 to get AVX in device code); it only takes effect at link time, where that code is generated.
  set(MOPTIM_NATIVE_CPU_ARCH "" CACHE STRING "-mcpu value for the native_cpu device code")
endif()
message(STATUS "moptim: SYCL backend ${MOPTIM_WITH_SYCL}")

add_library(moptim
  src/EigenSolver.cc
  src/ExecutionContext.cc
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc
  src/NormalEquations.cc)

if(MOPTIM_WITH_SYCL)
  string(REPLACE "," ";" MOPTIM_SYCL_TARGET_LIST "${SYCL_TARGETS}")
  set(MOPTIM_SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})

  # Eigen 3.4's NEON code crashes the native_cpu DEVICE front end (arm_neon.h name mangling), so in the
  # device pass of a -fsycl compilation Eigen runs scalar, while the host pass keeps its SIMD code.
  # The device-only define is given to the compiler (not set in a header), so it applies to every
  # translation unit that links moptim whatever its include order, and to every target of SYCL_TARGETS:
  # `-Xsycl-target-frontend=<triple>` passes an option to the device front end of that target only.
  # -ffp-contract=on (clang's default) fuses a*b+c only within one source expression (the frontend emits
  # llvm.fmuladd), so the result does not depend on where a model is inlined. `fast` lets the backend fuse
  # across statements, differently per inlining context (MOPTIM_FP_CONTRACT).
  # Fixed-size Eigen members are shared between the host and the device pass (a model is copied into
  # USM), so their layout must be identical: EIGEN_MAX_STATIC_ALIGN_BYTES=16 on both passes.
  # SHELL: keeps CMake from de-duplicating the repeated -Xsycl-target-frontend halves.
  set(MOPTIM_FP_CONTRACT "on" CACHE STRING
      "Value of -ffp-contract for moptim and its consumers (on, fast or off). 'fast' lets the backend fuse \
across statements depending on the inlining context, which breaks the bitwise cost invariant between kernels \
once models are inlined into several kernels (seen with Eigen 5); 'on' fuses within expressions only.")
  set(MOPTIM_DEVICE_ONLY_FLAGS "")
  foreach(triple IN LISTS MOPTIM_SYCL_TARGET_LIST)
    list(APPEND MOPTIM_DEVICE_ONLY_FLAGS "SHELL:-Xsycl-target-frontend=${triple} -DEIGEN_DONT_VECTORIZE")
    if(triple STREQUAL "native_cpu")
      # The native_cpu device front end also aborts (ItaniumMangle "unexpected Neon vector element type") on
      # any <arm_neon.h>, which OpenCV's and abseil's (protobuf's) headers include under __ARM_NEON on an
      # aarch64 host. Undefining it in the device pass only keeps them out; a no-op on x86 hosts.
      # With a CPU that has FP16 arithmetic (e.g. -mcpu=cortex-a76), Eigen stores Eigen::half as __fp16,
      # which the device front end rejects as a parameter type: hide the feature in the device pass too.
      list(APPEND MOPTIM_DEVICE_ONLY_FLAGS
        "SHELL:-Xsycl-target-frontend=${triple} -U__ARM_NEON"
        "SHELL:-Xsycl-target-frontend=${triple} -U__ARM_NEON__"
        "SHELL:-Xsycl-target-frontend=${triple} -U__ARM_FEATURE_FP16_SCALAR_ARITHMETIC"
        "SHELL:-Xsycl-target-frontend=${triple} -U__ARM_FEATURE_FP16_VECTOR_ARITHMETIC")
    endif()
  endforeach()
  target_compile_options(moptim PUBLIC ${MOPTIM_SYCL_FLAGS} -ffp-contract=${MOPTIM_FP_CONTRACT} ${MOPTIM_DEVICE_ONLY_FLAGS})
  target_compile_definitions(moptim PUBLIC EIGEN_MAX_STATIC_ALIGN_BYTES=16 MOPTIM_HAS_SYCL=1)
  target_link_options(moptim PUBLIC ${MOPTIM_SYCL_FLAGS} -ffp-contract=${MOPTIM_FP_CONTRACT})
  if(MOPTIM_NATIVE_CPU_ARCH)
    target_link_options(moptim PUBLIC "SHELL:-Xsycl-target-frontend=native_cpu -mcpu=${MOPTIM_NATIVE_CPU_ARCH}")
  endif()
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