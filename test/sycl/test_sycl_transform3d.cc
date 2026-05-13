#include <gtest/gtest.h>

#include "moptim/AnalyticalCost.h"
#include "AsyncConsoleLogger.hh"
#include "ConsoleLogger.hh"
#include "moptim/NumericalCostForwardEuler.h"
#include "moptim/NumericalCostSycl.h"
#include "Timer.hh"
#include "test_helper.hh"
#include "transform3d.hh"

using namespace moptim;

const double sycl_vs_cpu_tolerance = 1e-2;

INSTANTIATE_TEST_SUITE_P(TestTransform3D1MillionPoints, TestTransform3D, ::testing::Values(1'000'000));

TEST_P(TestTransform3D, SyclCost) {
  auto logger = std::make_shared<ConsoleLogger>();
  logger->log(ILog::Level::INFO, "3D-Transforming {} Points", GetParam());
  Timer t0;
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  const auto num_elements = pointcloud_.size();

  NumericalCostForwardEuler<Point3Distance, double> normal_cost(transformed_pointcloud_[0].data(),
                                                                pointcloud_[0].data(), num_elements, 3, 3, 6);

  NumericalCostSycl<double, Point3Distance> sycl_cost(
      logger, queue, std::span<const double>(transformed_pointcloud_[0].data(), num_elements * 3),
      std::span<const double>(pointcloud_[0].data(), num_elements * 3), 3, 3, 6, num_elements);

  double x0[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  t0.start();
  const auto cost_sum = normal_cost.computeCost(x0);
  auto stop = t0.stop();
  logger->log(ILog::Level::INFO, "Normal cost: {} took {} us", cost_sum, stop);

  t0.start();
  const auto sycl_cost_sum = sycl_cost.computeCost(x0);
  stop = t0.stop();
  logger->log(ILog::Level::INFO, "Sycl cost: {} took {} us", sycl_cost_sum, stop);

  EXPECT_NEAR(cost_sum, sycl_cost_sum, 1e-5);
  EXPECT_NEAR(cost_sum, 30000.000, 1e-5);
}

TEST_P(TestTransform3D, SyclJacobian) {
  auto logger = std::make_shared<ConsoleLogger>();
  Timer t0;
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  const auto num_elements = pointcloud_.size();

  NumericalCostForwardEuler<Point3Distance, double> normal_cost(transformed_pointcloud_[0].data(),
                                                                pointcloud_[0].data(), num_elements, 3, 3, 6);

  NumericalCostSycl<double, Point3Distance> sycl_cost(
      logger, queue, std::span<const double>(transformed_pointcloud_[0].data(), num_elements * 3),
      std::span<const double>(pointcloud_[0].data(), num_elements * 3), 3, 3, 6, num_elements);

  double x0[]{0.1, 0.1, 0.1, 0.0, 0.0, 0.0};

  Eigen::Matrix<double, 6, 6> num_jtj;
  Eigen::Matrix<double, 6, 1> num_jtb;
  double num_total = 0.0;

  t0.start();
  normal_cost.computeLinearSystem(x0, num_jtj.data(), num_jtb.data(), num_total);
  auto stop = t0.stop();
  logger->log(ILog::Level::INFO, "Normal cost jacobian: took {} us", stop);

  Eigen::Matrix<double, 6, 6> num_jtj_sycl;
  Eigen::Matrix<double, 6, 1> num_jtb_sycl;
  double num_total_sycl = 0.0;

  t0.start();
  sycl_cost.computeLinearSystem(x0, num_jtj_sycl.data(), num_jtb_sycl.data(), num_total_sycl);
  stop = t0.stop();
  logger->log(ILog::Level::INFO, "Sycl cost jacobian: took {} us", stop);

  std::cout << "normal vs sycl\n";
  std::cout << (num_jtj) << std::endl;
  std::cout << (num_jtj_sycl) << std::endl;

  compareMatrices(num_jtj_sycl, num_jtj, 10.0);
  compareMatrices(num_jtb_sycl, num_jtb, 10.0);

  EXPECT_NEAR(num_total, num_total_sycl, sycl_vs_cpu_tolerance);
}