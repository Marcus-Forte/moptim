#include <gtest/gtest.h>

#include <Cost.hh>
#include <GaussNewton.hh>
#include <iostream>

#include "test_models.hh"

// Square
Point2 point[] = {{1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0, 1.0}};
// Transformed square
Point2 point_tf[4];

Point2 transformPoint(const Point2& p, const Eigen::Matrix3d& T) {
  Eigen::Vector3d p_{p.x, p.y, 1.0};
  Eigen::Vector3d p_tf = T * p_;
  return {p_tf[0], p_tf[1]};
}

class Test2DTransform : public ::testing::Test {
 protected:
};

TEST(Test2DTransform, test_simple) {
  // x, y, theta
  Eigen::VectorXd x0{{0.1, 0.1, 0.1}};
  // Transform dataset
  Eigen::Matrix3d t;
  t << std::cos(x0[2]), -std::sin(x0[2]), x0[0], std::sin(x0[2]), std::cos(x0[2]), x0[1], 0, 0, 1;

  std::transform(point, std::end(point), point_tf, [&](const Point2& p) -> Point2 { return transformPoint(p, t); });

  auto cost = std::make_shared<Cost<Point2, Point2, Point2Distance>>(point, point_tf, 4, 3);
  GaussNewton optim;
  optim.addCost(cost);

  // Set initial guess
  x0[0] = 1.79;
  x0[1] = -2.3;
  x0[2] = -0.3;
  for (int i = 0; i < 5; ++i) {
    optim.step(x0);
  }

  EXPECT_NEAR(x0[0], 0.1, 0.01);
  EXPECT_NEAR(x0[1], 0.1, 0.01);
  EXPECT_NEAR(x0[2], 0.1, 0.01);
}