#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include <sycl/sycl.hpp>

#include "moptim/LevenbergMarquardt.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/NumericalCostSycl.hh"
#include "test_helper.hh"
#include "transform2d.hh"

using namespace moptim;

const double sycl_vs_cpu_tolerance = 1e-1;

TEST_F(TestTransform2D, SyclCostAndJacobian) {
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  const auto num_elements = pointcloud_.size();

  NumericalCostSycl<double, Point2Distance> num_cost_sycl(
      queue, std::span<const double>(transformed_pointcloud_[0].data(), transformed_pointcloud_.size() * 2),
      std::span<const double>(pointcloud_[0].data(), pointcloud_.size() * 2), 2, 2, 3, num_elements);

  NumericalCostForwardEuler<Point2Distance, double> num_cost(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                                             num_elements, 2, 2, 3);

  double x[]{0.0, 0.0, 0.0};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cost_result, 1e-5);

  // Jacobian
  Eigen::Matrix<double, 3, 3> jtj_sycl;
  Eigen::Matrix<double, 3, 1> jtb_sycl;
  double total_sycl = 0.0;

  Eigen::Matrix<double, 3, 3> jtj;
  Eigen::Matrix<double, 3, 1> jtb;
  double total = 0.0;

  auto start = std::chrono::steady_clock::now();
  num_cost_sycl.computeLinearSystem(x, jtj_sycl.data(), jtb_sycl.data(), total_sycl);
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Sycl cost jacobian: took " << elapsed << " us" << std::endl;

  start = std::chrono::steady_clock::now();
  num_cost.computeLinearSystem(x, jtj.data(), jtb.data(), total);
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Known cost jacobian: took " << elapsed << " us" << std::endl;

  std::cout << "num_jtj_sycl:\n" << jtj_sycl << " " << std::endl;
  std::cout << "num_jtj:\n" << jtj << " " << std::endl;

  EXPECT_NEAR(total_sycl, total, sycl_vs_cpu_tolerance);

  compareMatrices(jtj_sycl, jtj, sycl_vs_cpu_tolerance);
  compareMatrices(jtb_sycl, jtb, sycl_vs_cpu_tolerance);
}

TEST_F(TestTransform2D, Sycl2DTransformLM) {
  const auto num_elements = pointcloud_.size();

  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  auto solver = std::make_shared<LevenbergMarquardt<double>>(3);

  auto cost = std::make_shared<NumericalCostSycl<double, Point2Distance>>(
      queue, std::span<const double>(transformed_pointcloud_[0].data(), transformed_pointcloud_.size() * 2),
      std::span<const double>(pointcloud_[0].data(), pointcloud_.size() * 2), 2, 2, 3, num_elements);

  double x0[]{0, 0, 0};

  solver->addCost(cost);

  solver->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
}

// TODO
TEST_F(TestTransform2D, DISABLED_Sycl2DTransformLMAnalytical) {}
