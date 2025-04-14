#include <gtest/gtest.h>

#include <AnalyticalCost.hh>

#include "AnalyticalCost.hh"
#include "GaussNewton.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "test_models.hh"

using namespace test_models;

TEST(TestSimpleModel, GaussNewton) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<NumericalCost>(x_data_.data(), y_data_.data(), x_data_.size(), 1, model,
                                              DifferentiationMethod::CENTRAL);

  GaussNewton solver;

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, GaussNewtonAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost>(x_data_.data(), y_data_.data(), x_data_.size(), 1, model);
  GaussNewton solver;

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardt) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost>(x_data_.data(), y_data_.data(), x_data_.size(), 1, model);

  LevenbergMarquardt solver;

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestSimpleModel, LevenbergMarquardtAnalytical) {
  Eigen::VectorXd x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel>();
  auto cost = std::make_shared<AnalyticalCost>(x_data_.data(), y_data_.data(), x_data_.size(), 1, model);

  LevenbergMarquardt solver;

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}