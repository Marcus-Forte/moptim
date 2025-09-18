
add_compile_definitions(-DTEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
find_package(GTest REQUIRED)

add_library(test_helper
  transform2d.cc
  transform3d.cc
  test_helper.cc
)
target_link_libraries(test_helper moptim)
link_libraries(test_helper gtest)

add_executable(test_moptim
  test_main.cc
  test_simple_model.cc
  test_curve_fitting.cc
  test_powell.cc
  test_rosenbrock.cc
  test_transform2d.cc
  test_cost.cc
)
add_subdirectory(nist) # TODO fix
target_link_libraries(test_moptim test_helper)
gtest_discover_tests(test_moptim)

if(WITH_SYCL)
  add_executable(test_moptim_sycl
    sycl/test_sycl_cost.cc
    sycl/test_sycl_jacobian.cc
    # sycl/test_sycl_transform2d.cc
    # sycl/test_sycl_transform3d.cc
  )
  target_include_directories(test_moptim_sycl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_moptim_sycl gtest gtest_main)
  gtest_discover_tests(test_moptim_sycl)
  # Bug reproduction
  # add_executable(bug_reduction
  # sycl/bug_reduction.cc
  # )
  # add_executable(test_sycl_matmult
  #   sycl/test_sycl_matmult.cc
  # )
  # target_link_libraries(test_sycl_matmult onemath_blas_generic)
endif()


# Playground code for quick testing
add_executable(moptim_playground
  playground.cc
)
target_link_libraries(moptim_playground moptim)

add_executable(test_realtime test_realtime.cc)
target_link_libraries(test_realtime moptim)


