#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

TEST(TestSimpleModel, GaussNewton) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<NumericalCost<double>>(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model,
                                                      DifferentiationMethod::CENTRAL);

  GaussNewton<double> solver(2, std::make_shared<ConsoleLogger>(ILog::Level::INFO));

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, GaussNewtonAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost<double>>(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);
  GaussNewton<double> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardt) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost<double>>(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);

  LevenbergMarquardt<double> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardtAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost<double>>(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);

  LevenbergMarquardt<double> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}