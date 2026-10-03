#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "moptim/AnalyticalCost.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/NumericalCostSycl.hh"
#include "test_helper.hh"
#include "transform3d.hh"

using namespace moptim;

const double sycl_vs_cpu_tolerance = 1e-2;

INSTANTIATE_TEST_SUITE_P(TestTransform3D1MillionPoints, TestTransform3D, ::testing::Values(1'000'000));

TEST_P(TestTransform3D, SyclCost) {
  std::cout << "3D-Transforming " << GetParam() << " Points" << std::endl;
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  const auto num_elements = pointcloud_.size();

  NumericalCostForwardEuler<Point3Distance, double> normal_cost(transformed_pointcloud_[0].data(),
                                                                pointcloud_[0].data(), num_elements, 3, 3, 6);

  NumericalCostSycl<double, Point3Distance> sycl_cost(
      queue, std::span<const double>(transformed_pointcloud_[0].data(), num_elements * 3),
      std::span<const double>(pointcloud_[0].data(), num_elements * 3), 3, 3, 6, num_elements);

  double x0[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  auto start = std::chrono::steady_clock::now();
  const auto cost_sum = normal_cost.computeCost(x0);
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Normal cost: " << cost_sum << " took " << elapsed << " us" << std::endl;

  start = std::chrono::steady_clock::now();
  const auto sycl_cost_sum = sycl_cost.computeCost(x0);
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Sycl cost: " << sycl_cost_sum << " took " << elapsed << " us" << std::endl;

  EXPECT_NEAR(cost_sum, sycl_cost_sum, 1e-5);
  EXPECT_NEAR(cost_sum, 30000.000, 1e-5);
}

TEST_P(TestTransform3D, SyclJacobian) {
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  const auto num_elements = pointcloud_.size();

  NumericalCostForwardEuler<Point3Distance, double> normal_cost(transformed_pointcloud_[0].data(),
                                                                pointcloud_[0].data(), num_elements, 3, 3, 6);

  NumericalCostSycl<double, Point3Distance> sycl_cost(
      queue, std::span<const double>(transformed_pointcloud_[0].data(), num_elements * 3),
      std::span<const double>(pointcloud_[0].data(), num_elements * 3), 3, 3, 6, num_elements);

  double x0[]{0.1, 0.1, 0.1, 0.0, 0.0, 0.0};

  Eigen::Matrix<double, 6, 6> num_jtj;
  Eigen::Matrix<double, 6, 1> num_jtb;
  double num_total = 0.0;

  auto start = std::chrono::steady_clock::now();
  normal_cost.computeLinearSystem(x0, num_jtj.data(), num_jtb.data(), num_total);
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Normal cost jacobian: took " << elapsed << " us" << std::endl;

  Eigen::Matrix<double, 6, 6> num_jtj_sycl;
  Eigen::Matrix<double, 6, 1> num_jtb_sycl;
  double num_total_sycl = 0.0;

  start = std::chrono::steady_clock::now();
  sycl_cost.computeLinearSystem(x0, num_jtj_sycl.data(), num_jtb_sycl.data(), num_total_sycl);
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Sycl cost jacobian: took " << elapsed << " us" << std::endl;

  std::cout << "normal vs sycl\n";
  std::cout << (num_jtj) << std::endl;
  std::cout << (num_jtj_sycl) << std::endl;

  compareMatrices(num_jtj_sycl, num_jtj, 10.0);
  compareMatrices(num_jtb_sycl, num_jtb, 10.0);

  EXPECT_NEAR(num_total, num_total_sycl, sycl_vs_cpu_tolerance);
}
