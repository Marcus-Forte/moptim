#include <gtest/gtest.h>

#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostForwardEuler.hh"

using namespace moptim;

/**
 * @brief Model for the Rosenbrock function. No inputs or measurements, only parameters.
 *
 */
struct Rosenbrock : public IModel<double> {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
  }
  void f(const double* /*input*/, const double* /*measurement*/, double* f_x) override {
    f_x[0] = 10 * (x_[1] - x_[0] * x_[0]);
    f_x[1] = 1 - x_[0];
  }

  double x_[2];
};

TEST(TestRosenbrock, TestRosenbrock) {
  Eigen::VectorXd x{{3.0, -1.0}};

  const auto model = std::make_shared<Rosenbrock>();
  auto cost = std::make_shared<NumericalCostForwardEuler<double>>(x.data(), x.data(), 1, 2, 2, model);
  LevenbergMarquardt<double> solver(2, std::make_shared<ConsoleLogger>());
  solver.addCost(cost);

  solver.optimize(x.data());
  EXPECT_NEAR(x[0], 1.0, 1e-5);
  EXPECT_NEAR(x[1], 1.0, 1e-5);
}