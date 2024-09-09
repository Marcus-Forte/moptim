#include <gtest/gtest.h>

#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "NumericalCost.hh"

struct Powell {
  Powell(const Eigen::VectorXd& x0) : x_(x0) {}

  double operator()(double a, double b) { return 0; }
  Eigen::Vector4d operator()(double input, const Eigen::Vector4d& /*measurement*/) {
    const auto f0 = x_[0] + 10 * x_[1];
    const auto f1 = sqrt(5) * (x_[2] - x_[3]);
    const auto f2 = (x_[1] - 2 * x_[2]) * (x_[1] - 2 * x_[2]);
    const auto f3 = sqrt(10) * (x_[0] - x_[3]) * (x_[0] - x_[3]);

    return {f0, f1, f2, f3};
  }

  Eigen::VectorXd x_;
};

TEST(TestPowell, TestPowell) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};

  auto cost = std::make_shared<NumericalCost<double, Eigen::Vector4d, Powell>>(4);
  auto logger = std::make_shared<ConsoleLogger>();
  logger->setLevel(ILog::Level::INFO);
  GaussNewton solver(logger);
  solver.setMaxIterations(20);
  solver.addCost(cost);

  solver.optimize(x);
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);

  cost->getCost(x);
}