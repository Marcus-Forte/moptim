#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

TEST(TestCost, CostEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel<double>>();

  AnalyticalCost<double> an_cost(TestData<double>::x_data_, TestData<double>::y_data_, 1, 1, 2,
                                 TestData<double>::num_measurements, model);
  NumericalCostForwardEuler<double> num_cost(TestData<double>::x_data_, TestData<double>::y_data_, 1, 1, 2,
                                             TestData<double>::num_measurements, model);

  const auto an_cost_result = an_cost.computeCost(x.data());
  const auto num_cost_result = num_cost.computeCost(x.data());

  EXPECT_NEAR(an_cost_result, num_cost_result, 1e-5);
  EXPECT_NEAR(an_cost_result, 0.13670093591408203, 1e-5);
}

TEST(TestCost, JacobianEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel<double>>();

  const auto* x_data_ = TestData<double>::x_data_;
  const auto* y_data_ = TestData<double>::y_data_;
  const auto num_measurements = TestData<double>::num_measurements;

  AnalyticalCost<double> an_cost(x_data_, y_data_, 1, 1, 2, num_measurements, model);
  NumericalCostForwardEuler<double> num_cost(x_data_, y_data_, 1, 1, 2, num_measurements, model);
  Eigen::MatrixXd num_jtj(2, 2);
  Eigen::VectorXd num_jtb(2);
  double num_total = 0.0;

  Eigen::MatrixXd an_jtj(2, 2);
  Eigen::VectorXd an_jtb(2);
  double an_total = 0.0;

  an_cost.computeLinearSystem(x.data(), an_jtj.data(), an_jtb.data(), &an_total);
  num_cost.computeLinearSystem(x.data(), num_jtj.data(), num_jtb.data(), &num_total);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(an_jtj(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < an_jtb.size(); ++i) {
    EXPECT_NEAR(an_jtb(i), an_jtb(i), 1e-5);
  }

  EXPECT_NEAR(an_total, num_total, 1e-5);
}
