cmake_minimum_required(VERSION 3.20)
project(moptim)
set(CMAKE_CXX_STANDARD 20)

option(USE_CLANG_TIDY "Enable clang-tidy" ON)
option(FORMAT_CODE "Code formatter" OFF)
if(USE_CLANG_TIDY)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
else()
set(CMAKE_CXX_CLANG_TIDY "")
endif()

find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)

if(FORMAT_CODE)
  find_program(FORMATTER clang-format)
  if(FORMATTER)
    message(STATUS "Code formatter found.")
    file(GLOB_RECURSE SRC_FILES
      ${CMAKE_SOURCE_DIR}/src/*.cc
      ${CMAKE_SOURCE_DIR}/include/*.hh
      ${CMAKE_SOURCE_DIR}/test/*.cc
      ${CMAKE_SOURCE_DIR}/utils/*.cc
      ${CMAKE_SOURCE_DIR}/utils/*.hh)
    add_custom_target(code-format ALL
    clang-format -i ${SRC_FILES})
  endif()
endif()

add_subdirectory(utils)

add_library(moptim 
  src/GaussNewton.cc
  src/LevenbergMarquardt.cc)
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim utils Eigen3::Eigen tbb)

include(CTest)
if(BUILD_TESTING)
add_subdirectory(test)
endif()