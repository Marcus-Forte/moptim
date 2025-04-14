#include <gtest/gtest.h>

#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "transform3d.hh"

INSTANTIATE_TEST_SUITE_P(Test3DTransform10MillionPoints, Test3DTransform, ::testing::Values(1'000'000));

TEST_P(Test3DTransform, SyclCost) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};

  auto cost = std::make_shared<NumericalCostSycl<Eigen::Vector3d, Eigen::Vector3d, Point3Distance>>(
      &transformed_pointcloud_, &pointcloud_, queue);
  auto known_cost = std::make_shared<NumericalCost<Eigen::Vector3d, Eigen::Vector3d, Point3Distance>>(
      &transformed_pointcloud_, &pointcloud_);
  Eigen::VectorXd x0{{0.0, 0.0, 0, 0, 0, 0}};

  t0.start();
  const auto known_cost_sum = known_cost->computeCost(x0);
  auto stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Known cost: {} took {} us", known_cost_sum, stop);

  t0.start();
  const auto cost_sum = cost->computeCost(x0);
  stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Sycl cost: {} took {} us", cost_sum, stop);

  EXPECT_NEAR(cost_sum, known_cost_sum, 1e-5);
}

/// \todo fixme
TEST_P(Test3DTransform, DISABLED_SyclJacobian) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};

  auto cost = std::make_shared<NumericalCostSycl<Eigen::Vector3d, Eigen::Vector3d, Point3Distance>>(
      &transformed_pointcloud_, &pointcloud_, queue);
  auto known_cost = std::make_shared<NumericalCost<Eigen::Vector3d, Eigen::Vector3d, Point3Distance>>(
      &transformed_pointcloud_, &pointcloud_);
  Eigen::VectorXd x0{{0.0, 0.0, 0, 0, 0, 0}};

  t0.start();
  const auto [num_jtj, num_jtb, num_total] = known_cost->computeLinearSystem(x0);
  auto stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Known cost jacobian: took {} us", stop);

  t0.start();
  const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = cost->computeLinearSystem(x0);
  stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Sycl cost jacobian: took {} us", stop);

  for (int i = 0; i < num_jtj_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtj_sycl(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < num_jtb_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtb_sycl(i), num_jtb(i), 1e-5);
  }

  // EXPECT_NEAR(cost_sum, known_cost_sum, 1e-5);
}