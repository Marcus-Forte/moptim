#include <gtest/gtest.h>

#include "GaussNewton.hh"
#include "NumericalCost.hh"
#include "models.hh"

class TestSimpleModel : public ::testing::Test {
 protected:
  const std::vector<double> x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
  const std::vector<double> y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement
};

TEST_F(TestSimpleModel, cost) {
  Eigen::VectorXd x0{{0.9, 0.2}};
  NumericalCost<double, double, SimpleModel> cost(&x_data_, &y_data_, 2);
  const auto& jacobian = cost.computeJacobian(x0);
  const auto& [residual, total_cost] = cost.computeResidual(x0);

  const auto Jtj = jacobian.transpose() * jacobian;
  auto Jtb = jacobian.transpose() * residual;

  const auto& [Jtj_, Jtb_, total_error] = cost.computeHessian(x0);

  EXPECT_EQ(Jtj, Jtj_);
  EXPECT_EQ(Jtb, Jtb_);
}

TEST_F(TestSimpleModel, gauss_newton) {
  Eigen::VectorXd x{{0.9, 0.2}};
  GaussNewton solver;

  auto cost = std::make_shared<NumericalCost<double, double, SimpleModel>>(&x_data_, &y_data_, 2);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}
