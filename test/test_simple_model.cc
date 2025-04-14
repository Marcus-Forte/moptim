#include <gtest/gtest.h>

#include <AnalyticalCost.hh>

#include "GaussNewton.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "test_models.hh"

using namespace test_models;

TEST(TestSimpleModel, GaussNewton) {
  Eigen::VectorXd x{{0.9, 0.2}};

  GaussNewton solver;

  auto cost = std::make_shared<NumericalCost<double, double, SimpleModel, DifferentiationMethod::BACKWARD_EULER>>(
      &x_data_, &y_data_);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, GaussNewtonAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};

  GaussNewton solver;

  auto cost = std::make_shared<AnalyticalCost<double, double, SimpleModel>>(&x_data_, &y_data_);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardt) {
  Eigen::VectorXd x{{0.9, 0.2}};
  LevenbergMarquardt solver;

  auto cost = std::make_shared<NumericalCost<double, double, SimpleModel, DifferentiationMethod::BACKWARD_EULER>>(
      &x_data_, &y_data_);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardtAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};
  LevenbergMarquardt solver;

  auto cost = std::make_shared<AnalyticalCost<double, double, SimpleModel>>(&x_data_, &y_data_);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}