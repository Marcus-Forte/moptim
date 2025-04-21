#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "test_helper.hh"
#include "transform3d.hh"

const double sycl_vs_cpu_tolerance = 1e-2;

INSTANTIATE_TEST_SUITE_P(TestTransform3D1MillionPoints, TestTransform3D, ::testing::Values(1'000'000));

TEST_P(TestTransform3D, SyclCost) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  g_logging->log(ILog::Level::INFO, "3D-Transforming {} Points", GetParam());
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};

  const auto num_elements = pointcloud_.size();  // pointcloud_.size();

  const auto model = std::make_shared<Point3Distance>();

  auto normal_cost = std::make_shared<NumericalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                                     num_elements, 3, 6, model);

  auto sycl_cost = std::make_shared<NumericalCostSycl<Point3Distance>>(queue, transformed_pointcloud_[0].data(),
                                                                       pointcloud_[0].data(), num_elements, 3, 6);

  Eigen::VectorXd x0{{0.0, 0.0, 0, 0, 0, 0}};

  t0.start();
  const auto cost_sum = normal_cost->computeCost(x0);
  auto stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Known cost: {} took {} us", cost_sum, stop);

  t0.start();
  const auto sycl_cost_sum = sycl_cost->computeCost(x0);
  stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Sycl cost: {} took {} us", sycl_cost_sum, stop);

  EXPECT_NEAR(cost_sum, sycl_cost_sum, 1e-5);
  EXPECT_NEAR(cost_sum, 30000.000, 1e-5);
}

TEST_P(TestTransform3D, SyclJacobian) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};

  const auto num_elements = pointcloud_.size();

  const auto model = std::make_shared<Point3Distance>();

  auto normal_cost = std::make_shared<NumericalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                                     num_elements, 3, 6, model);

  auto sycl_cost = std::make_shared<NumericalCostSycl<Point3Distance>>(queue, transformed_pointcloud_[0].data(),
                                                                       pointcloud_[0].data(), num_elements, 3, 6);

  Eigen::VectorXd x0{{0.1, 0.1, 0.1, 0, 0, 0}};

  t0.start();
  const auto [num_jtj, num_jtb, num_total] = normal_cost->computeLinearSystem(x0);
  auto stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Known cost jacobian: took {} us", stop);

  t0.start();
  const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = sycl_cost->computeLinearSystem(x0);
  stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Sycl cost jacobian: took {} us", stop);

  std::cout << "normal vs sycl\n";
  std::cout << (num_jtj) << std::endl;
  std::cout << (num_jtj_sycl) << std::endl;

  compareMatrices(num_jtj_sycl, num_jtj, 10.0);
  compareMatrices(num_jtb_sycl, num_jtb, 10.0);

  EXPECT_NEAR(num_total, num_total_sycl, sycl_vs_cpu_tolerance);
}