#include <gtest/gtest.h>

#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"

struct Rosenbrock {
  Rosenbrock(const Eigen::VectorXd& x0) : x_(x0) {}

  Eigen::Vector2d operator()(double /*input*/, const Eigen::Vector2d& /* observation */) {
    const auto f0 = 10 * (x_[1] - x_[0] * x_[0]);
    const auto f1 = 1 - x_[0];

    return {f0, f1};
  }

  const Eigen::VectorXd x_;
};

TEST(TestRosenbrock, TestRosenbrock) {
  Eigen::VectorXd x{{3.0, -1.0}};
  auto cost = std::make_shared<NumericalCost<double, Eigen::Vector2d, Rosenbrock>>();
  LevenbergMarquardt solver;
  solver.addCost(cost);

  solver.optimize(x);
  EXPECT_NEAR(x[0], 1.0, 1e-5);
  EXPECT_NEAR(x[1], 1.0, 1e-5);
}