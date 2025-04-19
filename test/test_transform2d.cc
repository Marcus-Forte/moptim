#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "Timer.hh"
#include "transform2d.hh"

TEST_F(Transform2D, 2DTransformLM) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  t0.start();
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  const auto model = std::make_shared<Point2Distance>();
  auto cost = std::make_shared<NumericalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                              transformed_pointcloud_.size(), 2, 3, model);
  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  auto delta = t0.stop();
}

// FIXME
TEST_F(Transform2D, DISABLED_2DTransformLMAnalytical) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  const auto model = std::make_shared<Point2Distance>();
  auto cost = std::make_shared<AnalyticalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
                                               transformed_pointcloud_.size(), 2, 3, model);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}
