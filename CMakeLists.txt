cmake_minimum_required(VERSION 3.20)
project(moptim)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)

find_package(Eigen3 REQUIRED)
find_program(FORMATTER clang-format)
if(FORMATTER)
  message(STATUS "Code formatter found.")
  file(GLOB_RECURSE SRC_FILES
  ${CMAKE_SOURCE_DIR}/src/*.cc
  ${CMAKE_SOURCE_DIR}/include/*.hh
  ${CMAKE_SOURCE_DIR}/test/*.cc)
  message(STATUS "SRC: ${SRC_FILES}")
  add_custom_target(code-format ALL
  clang-format -i ${SRC_FILES})
endif()

add_library(moptim 
src/Cost.cc)
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim Eigen3::Eigen)

include(CTest)
if(BUILD_TESTING)
find_package(GTest REQUIRED)
add_executable(simple_test test/simple.cc)
add_executable(simple_point2 test/simple_point2.cc)
target_link_libraries(simple_test moptim gtest)
target_link_libraries(simple_point2 moptim gtest)
gtest_discover_tests(simple_test)
endif()