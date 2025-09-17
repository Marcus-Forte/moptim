#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

const std::vector<double> x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
const std::vector<double> y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement

TEST(TestCost, CostEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel<double>>();

  AnalyticalCost<double> an_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);
  NumericalCostForwardEuler<double> num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);

  const auto an_cost_result = an_cost.computeCost(x.data());
  const auto num_cost_result = num_cost.computeCost(x.data());

  EXPECT_NEAR(an_cost_result, num_cost_result, 1e-5);
  EXPECT_NEAR(an_cost_result, 0.13670093591408203, 1e-5);
}

TEST(TestCost, JacobianEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel<double>>();

  AnalyticalCost<double> an_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);
  NumericalCostForwardEuler<double> num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);
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
