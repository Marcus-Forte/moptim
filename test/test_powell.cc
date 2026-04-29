#include <gtest/gtest.h>

#include <memory>

#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "NumericalCostForwardEuler.hh"

using namespace moptim;

/**
 * @brief Model for the Powell function. No inputs or measurements, only parameters.
 *   param_dim = 4, obs_dim = 4, num_elements = 1.
 */
struct Powell {
  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = x[0] + 10 * x[1];
    res[1] = sqrt(5) * (x[2] - x[3]);
    res[2] = (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]);
    res[3] = sqrt(10) * (x[0] - x[3]) * (x[0] - x[3]);
  }

  void jacobian(const double* x, const double* /*input*/, const double* /*obs*/, double* jac) {
    jac[0] = 1;
    jac[1] = 0;
    jac[2] = 0;
    jac[3] = sqrt(10) * 2 * (x[0] - x[3]);

    // Df / dx1
    jac[4] = 10;
    jac[5] = 0;
    jac[6] = 2 * (x[1] + 2 * x[2]);
    jac[7] = 0;

    // Df / dx2
    jac[8] = 0;
    jac[9] = sqrt(5);
    jac[10] = 2 * (x[1] + 2 * x[2]) * (-2);
    jac[11] = 0;

    // Df / dx3
    jac[12] = 0;
    jac[13] = -sqrt(5);
    jac[14] = 0;
    jac[15] = sqrt(10) * 2 * (x[0] - x[3]) * (-1);
  }
};

TEST(TestPowell, TestPowell) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};
  const std::array<double, 4> input{0.0, 0.0, 0.0, 0.0};
  const std::array<double, 4> measurement{0.0, 0.0, 0.0, 0.0};

  auto cost = std::make_shared<NumericalCostForwardEuler<Powell, double>>(input.data(), measurement.data(), 1, 4, 4, 4);
  GaussNewton<double> solver(4, std::make_shared<ConsoleLogger>());
  solver.setMaxIterations(20);
  solver.addCost(cost);

  solver.optimize(x.data());
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);
}

struct PowellF0 {
  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = x[0] + 10 * x[1];
  }
};

struct PowellF1 {
  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = sqrt(5) * (x[2] - x[3]);
  }
};

struct PowellF2 {
  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]);
  }
};

struct PowellF3 {
  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = sqrt(10) * (x[0] - x[3]) * (x[0] - x[3]);
  }
};

// It is also possible to split a multi-dimensional function into multiple functions with a shared parameter set (x)
TEST(TestPowell, TestPowerllSplit) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};
  const std::array<double, 4> input{0.0, 0.0, 0.0, 0.0};
  const std::array<double, 1> measurement{0.0};

  auto cost1 =
      std::make_shared<NumericalCostForwardEuler<PowellF0, double>>(input.data(), measurement.data(), 1, 4, 1, 4);
  auto cost2 =
      std::make_shared<NumericalCostForwardEuler<PowellF1, double>>(input.data(), measurement.data(), 1, 4, 1, 4);
  auto cost3 =
      std::make_shared<NumericalCostForwardEuler<PowellF2, double>>(input.data(), measurement.data(), 1, 4, 1, 4);
  auto cost4 =
      std::make_shared<NumericalCostForwardEuler<PowellF3, double>>(input.data(), measurement.data(), 1, 4, 1, 4);

  auto logger = std::make_shared<ConsoleLogger>();
  logger->setLevel(ILog::Level::INFO);
  GaussNewton<double> solver(4, logger);
  solver.setMaxIterations(20);
  solver.addCost(cost1);
  solver.addCost(cost2);
  solver.addCost(cost3);
  solver.addCost(cost4);

  solver.optimize(x.data());
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);
}