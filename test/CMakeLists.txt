
add_compile_definitions(-DTEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_library(test_helper
  transform2d.cc
  transform3d.cc
  test_helper.cc
)
target_link_libraries(test_helper moptim gtest)

add_executable(test_moptim
  test_main.cc
  test_simple_model.cc
  test_curve_fitting.cc
  test_powell.cc
  test_rosenbrock.cc
  test_transform2d.cc
  test_transform3d.cc
  test_cost.cc
  test_active_param_dim.cc
  test_observer.cc
  test_loss_function.cc
  test_robust_cost.cc
  test_normal_equations.cc
  test_chunked_cost.cc
  test_optimizer.cc
  test_manifold.cc
  test_regressions.cc
  test_execution_context.cc
  test_device_loss.cc
  test_device_normal_equations.cc
  test_determinism.cc
  test_cost_models.cc
  test_tu_models.cc
  test_tu_models_a.cc
  test_tu_models_b.cc
)
add_subdirectory(nist)
target_link_libraries(test_moptim moptim test_helper)
gtest_discover_tests(test_moptim)

# Opt-in micro-benchmark of kernel launch latency and multi-core scaling of the default device.
add_executable(bench_launch bench_launch.cc)
target_link_libraries(bench_launch moptim)

# Benchmarks of the cost classes and optimizers. Built with the tests, run by hand.
add_executable(bench_moptim
  bench_moptim.cc
)
target_link_libraries(bench_moptim moptim)

# Playground code for quick testing
add_executable(moptim_playground
  playground.cc
)
target_link_libraries(moptim_playground moptim)
