#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCost.hh"
#include "test_models.hh"

using namespace test_models;

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
