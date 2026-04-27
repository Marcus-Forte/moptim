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
  void setup(std::span<const double> x) override {
    x_[0] = x[0];
    x_[1] = x[1];
  }
  void f(std::span<const double> /*input*/, std::span<const double> /*measurement*/, std::span<double> f_x) override {
    f_x[0] = 10 * (x_[1] - x_[0] * x_[0]);
    f_x[1] = 1 - x_[0];
  }

  double x_[2];
};

TEST(TestRosenbrock, TestRosenbrock) {
  Eigen::VectorXd x{{3.0, -1.0}};
  const std::array<double, 2> input{0.0, 0.0};
  const std::array<double, 2> measurement{0.0, 0.0};

  const auto model = std::make_shared<Rosenbrock>();
  auto cost = std::make_shared<NumericalCostForwardEuler<double>>(std::span<const double>(input),
                                                                  std::span<const double>(measurement), 2, 2, 2,
                                                                  model);
  LevenbergMarquardt<double> solver(2, std::make_shared<ConsoleLogger>());
  solver.addCost(cost);

  solver.optimize(std::span<double>(x.data(), x.size()));
  EXPECT_NEAR(x[0], 1.0, 1e-5);
  EXPECT_NEAR(x[1], 1.0, 1e-5);
}