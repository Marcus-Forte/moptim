#include <gtest/gtest.h>

#include "moptim/AnalyticalCost.h"
#include "ConsoleLogger.hh"
#include "moptim/LevenbergMarquardt.h"
#include "moptim/NumericalCostForwardEuler.h"
#include "Timer.hh"
#include "transform2d.hh"

using namespace moptim;

TEST_F(TestTransform2D, 2DTransformLM) {
  auto logger = std::make_shared<ConsoleLogger>();
  Timer t0;
  t0.start();
  auto solver = std::make_shared<LevenbergMarquardt<double>>(3, logger);
  const auto model = std::make_shared<Point2Distance>();
  auto cost = std::make_shared<NumericalCostForwardEuler<Point2Distance, double>>(
      transformed_pointcloud_[0].data(), pointcloud_[0].data(), transformed_pointcloud_.size(), 2, 2, 3);
  solver->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver->optimize(x0.data());

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
}

TEST_F(TestTransform2D, 2DTransformLMAnalytical) {
  auto logger = std::make_shared<ConsoleLogger>();
  auto solver = std::make_shared<LevenbergMarquardt<double>>(3, logger);
  const auto model = std::make_shared<Point2Distance>();
  auto cost = std::make_shared<AnalyticalCost<Point2Distance, double>>(
      transformed_pointcloud_[0].data(), pointcloud_[0].data(), transformed_pointcloud_.size(), 2, 2, 3);

  solver->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver->optimize(x0.data());

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
}