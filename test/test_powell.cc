#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "NumericalCost.hh"

struct Powell {
  Powell(const Eigen::VectorXd& x0) : x_(x0) {}

  Eigen::Vector4d operator()(double /*input*/, const Eigen::Vector4d& /* observation */) {
    const auto f0 = x_[0] + 10 * x_[1];
    const auto f1 = sqrt(5) * (x_[2] - x_[3]);
    const auto f2 = (x_[1] - 2 * x_[2]) * (x_[1] - 2 * x_[2]);
    const auto f3 = sqrt(10) * (x_[0] - x_[3]) * (x_[0] - x_[3]);

    return {f0, f1, f2, f3};
  }

  Eigen::Matrix<double, 4, 4> jacobian(double /* input */, const Eigen::Vector4d& /* observation */) {
    Eigen::Matrix4d jac;
    // Df / dx0
    jac(0) = 1;
    jac(4) = 0;
    jac(8) = 0;
    jac(12) = sqrt(10) * 2 * (x_[0] - x_[3]);

    // Df / dx1
    jac(1) = 10;
    jac(5) = 0;
    jac(9) = 2 * (x_[1] + 2 * x_[2]);
    jac(13) = 0;

    // Df / dx2
    jac(2) = 0;
    jac(6) = sqrt(5);
    jac(10) = 2 * (x_[1] + 2 * x_[2]) * (-2);
    jac(14) = 0;

    // Df / dx3
    jac(3) = 0;
    jac(7) = -sqrt(5);
    jac(11) = 0;
    jac(15) = sqrt(10) * 2 * (x_[0] - x_[3]) * (-1);
    return jac;
  }

  const Eigen::VectorXd x_;
};

TEST(TestPowell, TestPowell) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};

  auto cost = std::make_shared<NumericalCost<double, Eigen::Vector4d, Powell>>();
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
}

struct PowellF0 {
  PowellF0(const Eigen::VectorXd& x0) : x_(x0) {}
  double operator()(double /*input*/, double /*input*/) { return x_[0] + 10 * x_[1]; }
  const Eigen::VectorXd x_;
};

struct PowellF1 {
  PowellF1(const Eigen::VectorXd& x0) : x_(x0) {}
  double operator()(double /*input*/, double /*input*/) { return sqrt(5) * (x_[2] - x_[3]); }
  const Eigen::VectorXd x_;
};

struct PowellF2 {
  PowellF2(const Eigen::VectorXd& x0) : x_(x0) {}
  double operator()(double /*input*/, double /*input*/) { return (x_[1] - 2 * x_[2]) * (x_[1] - 2 * x_[2]); }
  const Eigen::VectorXd x_;
};

struct PowellF3 {
  PowellF3(const Eigen::VectorXd& x0) : x_(x0) {}
  double operator()(double /*input*/, double /*input*/) { return sqrt(10) * (x_[0] - x_[3]) * (x_[0] - x_[3]); }
  const Eigen::VectorXd x_;
};

// It is also possible to split a multi-dimensional function into multiple functions.
TEST(TestPowell, TestPowerllSplit) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};

  auto cost1 = std::make_shared<NumericalCost<double, double, PowellF0>>();
  auto cost2 = std::make_shared<NumericalCost<double, double, PowellF1>>();
  auto cost3 = std::make_shared<NumericalCost<double, double, PowellF2>>();
  auto cost4 = std::make_shared<NumericalCost<double, double, PowellF3>>();

  auto logger = std::make_shared<ConsoleLogger>();
  logger->setLevel(ILog::Level::INFO);
  GaussNewton solver(logger);
  solver.setMaxIterations(20);
  solver.addCost(cost1);
  solver.addCost(cost2);
  solver.addCost(cost3);
  solver.addCost(cost4);

  solver.optimize(x);
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);
}