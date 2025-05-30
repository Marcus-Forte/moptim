#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "test_helper.hh"
#include "transform2d.hh"

const double sycl_vs_cpu_tolerance = 1e-1;

TEST_F(TestTransform2D, SyclCostAndJacobian) {
  sycl::queue queue{sycl::default_selector_v};

  const auto num_elements = pointcloud_.size();

  const auto model = std::make_shared<Point2Distance>();

  NumericalCostSycl<Point2Distance> num_cost_sycl(queue, transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                                  num_elements, 2, 3);
  NumericalCost num_cost(transformed_pointcloud_[0].data(), pointcloud_[0].data(), num_elements, 2, 3, model);

  Eigen::VectorXd x{{0.0, 0.0, 0.0}};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cost_result, 1e-5);

  // Jacobian
  Timer t0;
  t0.start();

  const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = num_cost_sycl.computeLinearSystem(x);
  auto stop = t0.stop();
  std::cout << "Sycl cost jacobian: took " << stop << " us" << std::endl;

  t0.start();
  const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);
  stop = t0.stop();
  std::cout << "Known cost jacobian: took " << stop << " us" << std::endl;

  std::cout << "num_jtj_sycl:\n" << num_jtj_sycl << " " << std::endl;
  std::cout << "num_jtj:\n" << num_jtj << " " << std::endl;

  EXPECT_NEAR(num_total_sycl, num_total, sycl_vs_cpu_tolerance);

  compareMatrices(num_jtj_sycl, num_jtj, sycl_vs_cpu_tolerance);
  compareMatrices(num_jtb_sycl, num_jtb, sycl_vs_cpu_tolerance);
}

TEST_F(TestTransform2D, Sycl2DTransformLM) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  const auto num_elements = pointcloud_.size();

  Timer t0;
  t0.start();
  sycl::queue queue{sycl::default_selector_v};
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);

  auto cost = std::make_shared<NumericalCostSycl<Point2Distance>>(queue, transformed_pointcloud_[0].data(),
                                                                  pointcloud_[0].data(), num_elements, 2, 3);

  auto model = std::make_shared<Point2Distance>();
  Eigen::VectorXd x0{{0, 0, 0}};
  const auto [jtj, jtb, res] = cost->computeLinearSystem(x0);

  solver_->addCost(cost);

  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  auto delta = t0.stop();
}

// FIXME
// TEST_F(Transform2D, DISABLED_Sycl2DTransformLMAnalytical) {
//   auto g_logging = std::make_shared<ConsoleLogger>();
//   solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
//   const auto model = std::make_shared<Point2Distance>();
//   auto cost = std::make_shared<AnalyticalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
//                                                transformed_pointcloud_.size(), 2, model);

//   solver_->addCost(cost);
//   Eigen::VectorXd x0{{0, 0, 0}};
//   solver_->optimize(x0);

//   EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
//   EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
//   EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
// }
