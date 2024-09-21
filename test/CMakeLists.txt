
add_compile_definitions(-DTEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
find_package(GTest REQUIRED)
add_executable(moptim_tests
  test_main.cc
  test_simple_model.cc
  test_curve_fitting.cc
  test_powell.cc
  test_rosenbrock.cc
  test_2dtransform.cc
  test_cost.cc
)
add_subdirectory(nist)
target_link_libraries(moptim_tests moptim gtest)
gtest_discover_tests(moptim_tests)