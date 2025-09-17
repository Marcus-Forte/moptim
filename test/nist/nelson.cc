#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <LevenbergMarquardt.hh>

#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"

using namespace moptim;

const static double y_data[]{
    15.00E0, 17.00E0, 15.50E0, 16.50E0, 15.50E0, 15.00E0, 16.00E0, 14.50E0, 15.00E0, 14.50E0, 12.50E0, 11.00E0, 14.00E0,
    13.00E0, 14.00E0, 11.50E0, 14.00E0, 16.00E0, 13.00E0, 13.50E0, 13.00E0, 13.50E0, 12.50E0, 12.50E0, 12.50E0, 12.00E0,
    11.50E0, 12.00E0, 13.00E0, 11.50E0, 13.00E0, 12.50E0, 13.50E0, 17.50E0, 17.50E0, 13.50E0, 12.50E0, 12.50E0, 15.00E0,
    13.00E0, 12.00E0, 13.00E0, 12.00E0, 13.50E0, 10.00E0, 11.50E0, 11.00E0, 9.50E0,  15.00E0, 15.00E0, 15.50E0, 16.00E0,
    13.00E0, 10.50E0, 13.50E0, 14.00E0, 12.50E0, 12.00E0, 11.50E0, 11.50E0, 6.50E0,  5.50E0,  6.00E0,  6.00E0,  18.50E0,
    17.00E0, 15.30E0, 16.00E0, 13.00E0, 14.00E0, 12.50E0, 11.00E0, 12.00E0, 12.00E0, 11.50E0, 12.00E0, 6.00E0,  6.00E0,
    5.00E0,  5.50E0,  12.50E0, 13.00E0, 16.00E0, 12.00E0, 11.00E0, 9.50E0,  11.00E0, 11.00E0, 11.00E0, 10.00E0, 10.50E0,
    10.50E0, 2.70E0,  2.70E0,  2.50E0,  2.40E0,  13.00E0, 13.50E0, 16.50E0, 13.60E0, 11.50E0, 10.50E0, 13.50E0, 12.00E0,
    7.00E0,  6.90E0,  8.80E0,  7.90E0,  1.20E0,  1.50E0,  1.00E0,  1.50E0,  13.00E0, 12.50E0, 16.50E0, 16.00E0, 11.00E0,
    11.50E0, 10.50E0, 10.00E0, 7.27E0,  7.50E0,  6.70E0,  7.60E0,  1.50E0,  1.00E0,  1.20E0,  1.20E0,
};

const static double x_data[]{
    1E0,  180E0, 1E0,  180E0, 1E0,  180E0, 1E0,  180E0, 1E0,  225E0, 1E0,  225E0, 1E0,  225E0, 1E0,  225E0, 1E0,  250E0,
    1E0,  250E0, 1E0,  250E0, 1E0,  250E0, 1E0,  275E0, 1E0,  275E0, 1E0,  275E0, 1E0,  275E0, 2E0,  180E0, 2E0,  180E0,
    2E0,  180E0, 2E0,  180E0, 2E0,  225E0, 2E0,  225E0, 2E0,  225E0, 2E0,  225E0, 2E0,  250E0, 2E0,  250E0, 2E0,  250E0,
    2E0,  250E0, 2E0,  275E0, 2E0,  275E0, 2E0,  275E0, 2E0,  275E0, 4E0,  180E0, 4E0,  180E0, 4E0,  180E0, 4E0,  180E0,
    4E0,  225E0, 4E0,  225E0, 4E0,  225E0, 4E0,  225E0, 4E0,  250E0, 4E0,  250E0, 4E0,  250E0, 4E0,  250E0, 4E0,  275E0,
    4E0,  275E0, 4E0,  275E0, 4E0,  275E0, 8E0,  180E0, 8E0,  180E0, 8E0,  180E0, 8E0,  180E0, 8E0,  225E0, 8E0,  225E0,
    8E0,  225E0, 8E0,  225E0, 8E0,  250E0, 8E0,  250E0, 8E0,  250E0, 8E0,  250E0, 8E0,  275E0, 8E0,  275E0, 8E0,  275E0,
    8E0,  275E0, 16E0, 180E0, 16E0, 180E0, 16E0, 180E0, 16E0, 180E0, 16E0, 225E0, 16E0, 225E0, 16E0, 225E0, 16E0, 225E0,
    16E0, 250E0, 16E0, 250E0, 16E0, 250E0, 16E0, 250E0, 16E0, 275E0, 16E0, 275E0, 16E0, 275E0, 16E0, 275E0, 32E0, 180E0,
    32E0, 180E0, 32E0, 180E0, 32E0, 180E0, 32E0, 225E0, 32E0, 225E0, 32E0, 225E0, 32E0, 225E0, 32E0, 250E0, 32E0, 250E0,
    32E0, 250E0, 32E0, 250E0, 32E0, 275E0, 32E0, 275E0, 32E0, 275E0, 32E0, 275E0, 48E0, 180E0, 48E0, 180E0, 48E0, 180E0,
    48E0, 180E0, 48E0, 225E0, 48E0, 225E0, 48E0, 225E0, 48E0, 225E0, 48E0, 250E0, 48E0, 250E0, 48E0, 250E0, 48E0, 250E0,
    48E0, 275E0, 48E0, 275E0, 48E0, 275E0, 48E0, 275E0, 64E0, 180E0, 64E0, 180E0, 64E0, 180E0, 64E0, 180E0, 64E0, 225E0,
    64E0, 225E0, 64E0, 225E0, 64E0, 225E0, 64E0, 250E0, 64E0, 250E0, 64E0, 250E0, 64E0, 250E0, 64E0, 275E0, 64E0, 275E0,
    64E0, 275E0, 64E0, 275E0};

struct Model : public IModel<double> {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
  }

  void f(const double* input, const double* measurement, double* f_x) override {
    const auto f = x_[0] - x_[1] * input[0] * std::exp(-x_[2] * input[1]);
    f_x[0] = measurement[0] - f;
  }
  double x_[3];
};

// TODO
TEST(nelson, nelson) {
  double x0[3]{2, 0.0001, -0.01};
  const auto model = std::make_shared<Model>();
  auto cost =
      std::make_shared<NumericalCostForwardEuler<double>>(x_data, y_data, sizeof(x_data) / sizeof(double), 1, 3, model);
  const auto logger = std::make_shared<ConsoleLogger>();
  logger->setLevel(ILog::Level::DEBUG);
  LevenbergMarquardt<double> solver(3, logger);
  solver.setMaxIterations(100);
  solver.addCost(cost);

  solver.optimize(x0);

  EXPECT_NEAR(x0[0], 2.5906836021E+00, 1e-3);
  // EXPECT_NEAR(x0[1], 6.1314004477E-03, 1e-3);
  // EXPECT_NEAR(x0[2], 1.0530908399E-02, 1e-3);
}