#include <gtest/gtest.h>

#include "ConsoleLogger.hh"
#include "moptim/LevenbergMarquardt.h"
#include "moptim/NumericalCostForwardEuler.h"

using namespace moptim;

/**
 * @brief Model for the Rosenbrock function. No inputs or measurements, only parameters.
 *
 */
struct Rosenbrock {
  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* /*input*/, const double* /*obs*/, double* res) {
    res[0] = 10 * (x[1] - x[0] * x[0]);
    res[1] = 1 - x[0];
  }
};

TEST(TestRosenbrock, TestRosenbrock) {
  Eigen::VectorXd x{{3.0, -1.0}};
  const std::array<double, 2> input{0.0, 0.0};
  const std::array<double, 2> measurement{0.0, 0.0};

  auto cost = std::make_shared<NumericalCostForwardEuler<Rosenbrock, double>>(input.data(), measurement.data(), 1, 2, 2,
                                                                              2, Rosenbrock{});
  LevenbergMarquardt<double> solver(2, std::make_shared<ConsoleLogger>());
  solver.addCost(cost);

  solver.optimize(x.data());
  EXPECT_NEAR(x[0], 1.0, 1e-5);
  EXPECT_NEAR(x[1], 1.0, 1e-5);
}