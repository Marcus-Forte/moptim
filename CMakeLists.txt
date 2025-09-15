cmake_minimum_required(VERSION 3.20)
project(moptim)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

option(USE_CLANG_TIDY "Enable clang-tidy" OFF)
option(WITH_SYCL "Use SYCL compilation for parallel computation" OFF)
add_compile_definitions(EIGEN_ARM64_USE_SVE EIGEN_DONT_VECTORIZE)

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
  find_package(oneMath  REQUIRED)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
  if(NOT COMPILER_SUPPORTS_FSYCL)
      message(FATAL_ERROR "SYCL compilation requires dpcpp compiler")
  endif()
  set(SYCL_FLAGS -fsycl -fsycl-targets=native_cpu,nvptx64-nvidia-cuda,spir64)
  target_compile_options(moptim PUBLIC ${SYCL_FLAGS})
  target_link_options(moptim PUBLIC ${SYCL_FLAGS})
  link_libraries(ONEMATH::onemath onemath_blas_generic) # The onemath library must have been compiled in the base container.
  # If `onemath_blas_cublas` is used, make sure to link `cuda` as well.
endif()
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim utils Eigen3::Eigen)

## Testing
if(BUILD_TESTING)
add_subdirectory(test)
endif()