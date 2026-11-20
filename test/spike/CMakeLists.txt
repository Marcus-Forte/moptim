# Phase 0 spike of the SYCL rewrite: standalone project, deliberately not part of the moptim build.
#
#   cmake -S test/spike -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++
#
# Needs only Eigen3 and a DPC++ with the native_cpu target. The same sources are built into four
# variants that differ in compile flags only (see add_spike() calls below).
cmake_minimum_required(VERSION 3.20)
project(moptim_spike CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)

# Host MCU for the spike_chunk_mcpu variant. The container reports an Apple core the kernel cannot
# name, so `-mcpu=native` resolves to "generic"; pick the real core explicitly.
set(SPIKE_MCPU apple-m3 CACHE STRING "-mcpu value of the spike_chunk_mcpu variant")

set(SYCL_FLAGS -fsycl -fsycl-targets=native_cpu)

# add_spike(<name> [DEFINITIONS ...] [FLAGS ...] [LINK_FLAGS ...] [EIGEN_DIR dir]): FLAGS go to both compile and link. Option pairs
# need the SHELL: prefix, or CMake de-duplicates the repeated halves.
function(add_spike name)
  cmake_parse_arguments(ARG "" "EIGEN_DIR" "DEFINITIONS;FLAGS;LINK_FLAGS" ${ARGN})
  add_executable(${name} spike_main.cc spike_second_tu.cc)
  if(ARG_EIGEN_DIR)
    target_include_directories(${name} SYSTEM PRIVATE ${ARG_EIGEN_DIR})
  else()
    target_link_libraries(${name} PRIVATE Eigen3::Eigen)
  endif()
  target_compile_options(${name} PRIVATE ${SYCL_FLAGS} ${ARG_FLAGS} -Wall -Wextra)
  target_link_options(${name} PRIVATE ${SYCL_FLAGS} ${ARG_FLAGS} ${ARG_LINK_FLAGS})
  target_compile_definitions(${name} PRIVATE ${ARG_DEFINITIONS})
endfunction()

# Default: Eigen vectorization on (NEON), Vecz on.
add_spike(spike_chunk)
# Vecz off through the device flag, which applies to every kernel of the binary.
add_spike(spike_chunk_novecz FLAGS "SHELL:-mllvm -sycl-native-cpu-no-vecz")
# Device and host code compiled for a specific core. The device flag only takes effect at link time
# (the native_cpu code is generated there; the compile step warns that it is unused). Eigen 3.4.0 does
# not compile with clang once the core has fp16 arithmetic (numext::bit_cast<__fp16> in Half.h,
# vmulh_f16 in NEON/PacketMath.h), hence the -U / fp16 override.
set(MCPU_FLAGS -mcpu=${SPIKE_MCPU} -U__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
set(MCPU_LINK "SHELL:-Xsycl-target-frontend=native_cpu -mcpu=${SPIKE_MCPU}")
set(FP16_DEFS EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC=0)


# Eigen as in the current SYCL build of moptim: no vectorization, SVE flag set.
add_spike(spike_chunk_eigen_scalar DEFINITIONS EIGEN_DONT_VECTORIZE EIGEN_ARM64_USE_SVE)

# Contraction of a*b+c into fma across statements for host and device (GCC's default, and so how the
# gcc baseline is compiled). Clang's default is -ffp-contract=on (within one expression only).
add_spike(spike_chunk_fast FLAGS -ffp-contract=fast)
add_spike(spike_chunk_fast_mcpu FLAGS -ffp-contract=fast ${MCPU_FLAGS} LINK_FLAGS ${MCPU_LINK} DEFINITIONS ${FP16_DEFS})
add_spike(spike_chunk_eigen_scalar_fast FLAGS -ffp-contract=fast DEFINITIONS EIGEN_DONT_VECTORIZE EIGEN_ARM64_USE_SVE)

# Same alignment of the model in the host and device pass (scalar Eigen would otherwise align to 8).
add_spike(spike_chunk_align16 DEFINITIONS EIGEN_MAX_STATIC_ALIGN_BYTES=16 SPIKE_EXPECTED_ALIGN=16)

# Vectorized Eigen in the device pass without arm_neon.h: the generic clang-vector packets of Eigen
# master (post 5.0; not in 3.4 or 5.0). Point SPIKE_EIGEN_MASTER at a checkout to enable the variant.
set(SPIKE_EIGEN_MASTER "" CACHE PATH "Eigen master checkout (for spike_chunk_generic)")
if(SPIKE_EIGEN_MASTER)
  add_spike(spike_chunk_generic FLAGS -ffp-contract=fast EIGEN_DIR ${SPIKE_EIGEN_MASTER}
            DEFINITIONS EIGEN_VECTORIZE_GENERIC EIGEN_GENERIC_VECTOR_SIZE_BYTES=16 EIGEN_HAS_ARM64_FP16=0)
endif()
