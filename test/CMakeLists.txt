
add_compile_definitions(-DTEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
find_package(GTest REQUIRED)

add_library(test_models
  test_models.cc
  transform2d.cc
  transform3d.cc
)
target_link_libraries(test_models moptim)
link_libraries(test_models)

add_executable(moptim_tests
  test_main.cc
  test_simple_model.cc
  test_curve_fitting.cc
  test_powell.cc
  test_rosenbrock.cc
  test_transform2d.cc
  test_cost.cc
)
# add_subdirectory(nist) # TODO fix
target_link_libraries(moptim_tests gtest)
gtest_discover_tests(moptim_tests)

add_executable(moptim_sycl_tests
  sycl/test_sycl_cost.cc
  sycl/test_sycl_transform3d.cc
)
target_include_directories(moptim_sycl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moptim_sycl_tests gtest gtest_main)
gtest_discover_tests(moptim_sycl_tests)