#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "NumericalCost2.hh"
#include "Timer.hh"
#include "transform2d.hh"

struct Point2Distance2 : public IModel {
  void setup(const double* x) override {
    transform_.setIdentity();
    transform_.rotate(x[2]);
    transform_.translate(Eigen::Vector2d{x[0], x[1]});
  }

  void f(const double* input, const double* measurement, double* f_x) override {
    const auto* target = reinterpret_cast<const Eigen::Vector2d*>(measurement);
    const auto* source = reinterpret_cast<const Eigen::Vector2d*>(input);
    auto* transformed_point = reinterpret_cast<Eigen::Vector2d*>(f_x);

    *transformed_point = *target - transform_ * (*source);
  }

  Eigen::Affine2d transform_;
};

TEST_F(Transform2D, 2DTransformLM) {
  auto g_logging = std::make_shared<ConsoleLogger>();
  Timer t0;
  t0.start();
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  auto cost = std::make_shared<
      NumericalCost<Eigen::Vector2d, Eigen::Vector2d, Point2Distance, DifferentiationMethod::BACKWARD_EULER>>(
      &transformed_pointcloud_, &pointcloud_);
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
  auto cost = std::make_shared<AnalyticalCost<Eigen::Vector2d, Eigen::Vector2d, Point2Distance>>(
      &transformed_pointcloud_, &pointcloud_);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}
