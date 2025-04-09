#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "models.hh"

const std::vector<double> x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
const std::vector<double> y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement

TEST(TestCost, JacobianEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  AnalyticalCost<double, double, SimpleModel> an_cost(&x_data_, &y_data_);
  NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_);

  const auto [an_jtj, an_jtb, an_total] = an_cost.computeLinearSystem(x);
  const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(an_jtj(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < an_jtb.size(); ++i) {
    EXPECT_NEAR(an_jtb(i), an_jtb(i), 1e-5);
  }

  EXPECT_NEAR(an_total, num_total, 1e-5);

  ConsoleLogger logg;
  logg.log(ILog::Level::INFO, "number: {}", 25);
}

TEST(TestCost, NumericalCostSycl) {
  NumericalCostSycl<double, double, SimpleModel> num_cost_sycl(&x_data_, &y_data_);
  NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_);

  Eigen::VectorXd x{{0.1, 0.1}};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_EQ(sycl_cost_result, cpu_cost_result);
}