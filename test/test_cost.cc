#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "NumericalCost.hh"
#include "models.hh"

Eigen::Vector2d setVals() { return {1, 2}; }

const std::vector<double> x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
const std::vector<double> y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement

TEST(Jacobian, Jacobian) {
  Eigen::VectorXd x0{{0.1, 0.1}};

  AnalyticalCost<double, double, SimpleModel> an_cost(&x_data_, &y_data_, 2);
  NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_, 2);

  const auto an_jac = an_cost.computeJacobian(x0);
  const auto num_jac = num_cost.computeJacobian(x0);

  for (int i = 0; i < an_jac.size(); ++i) {
    EXPECT_NEAR(an_jac(i), num_jac(i), 1e-6);
  }
}