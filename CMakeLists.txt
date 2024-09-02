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
    ${CMAKE_SOURCE_DIR}/test/*.cc
    ${CMAKE_SOURCE_DIR}/utils/*.cc
    ${CMAKE_SOURCE_DIR}/utils/*.hh)
  message(STATUS "SRC: ${SRC_FILES}")
  add_custom_target(code-format ALL
  clang-format -i ${SRC_FILES})
endif()

add_subdirectory(utils)

add_library(moptim 
  src/Cost.cc
  src/GaussNewton.cc)
target_include_directories(moptim PUBLIC include)
target_link_libraries(moptim utils Eigen3::Eigen)

include(CTest)
if(BUILD_TESTING)
find_package(GTest REQUIRED)
add_executable(moptim_tests
  test/test_main.cc
  test/test_simple_model.cc
  test/test_models.cc
  test/test_2dtransform.cc)
target_link_libraries(moptim_tests moptim gtest)
gtest_discover_tests(moptim_tests)
endif()