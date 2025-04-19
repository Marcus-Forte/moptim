#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "transform2d.hh"

TEST_F(Transform2D, SyclCost) {
  sycl::queue queue{sycl::default_selector_v};

  const auto num_elements = 5;  // pointcloud_.size();

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

  EXPECT_NEAR(num_total_sycl, num_total, 1e-5);

  for (int i = 0; i < num_jtj_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtj_sycl(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < num_jtb_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtb_sycl(i), num_jtb(i), 1e-5);
  }
}

TEST_F(Transform2D, DISABLED_Sycl2DTransformLM) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  t0.start();
  sycl::queue queue{sycl::default_selector_v};
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  auto cost = std::make_shared<NumericalCostSycl<Point2Distance>>(queue, transformed_pointcloud_[0].data(),
                                                                  pointcloud_[0].data(), 10, 2, 3);

  auto model = std::make_shared<Point2Distance>();
  auto cost_normal =
      std::make_shared<NumericalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(), 10, 2, 3, model);
  Eigen::VectorXd x0{{0, 0, 0}};
  const auto [jtj, jtb, res] = cost->computeLinearSystem(x0);
  const auto [jtj_normal, jtb_normal, res_normal] = cost_normal->computeLinearSystem(x0);
  std::cout << "sycl:" << jtj << std::endl;
  std::cout << "normal:" << jtj_normal << std::endl;
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
