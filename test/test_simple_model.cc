#include <gtest/gtest.h>

#include "Cost.hh"
#include "GaussNewton.hh"
#include "test_models.hh"

class TestSimpleModel : public ::testing::Test {
 protected:
  const double x_data_[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
  const double y_data_[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement
};

TEST_F(TestSimpleModel, cost) {
  Eigen::VectorXd x0{{0.9, 0.2}};

  Cost<double, double, SimpleModel> cost(x_data_, y_data_, 7, 2);

  const auto jacobian = cost.computeJacobian(x0);
  const auto residual = cost.computeResidual(x0);

  const auto jtj = jacobian.transpose() * jacobian;
  auto Jtb = jacobian.transpose() * residual;

  const auto& [jtj_, Jtb_] = cost.computeHessian(x0);

  EXPECT_EQ(jtj, jtj_);
  EXPECT_EQ(Jtb, Jtb_);
}

TEST_F(TestSimpleModel, gauss_newton) {
  Eigen::VectorXd x0{{0.9, 0.2}};
  GaussNewton optimizer;

  auto cost = std::make_shared<Cost<double, double, SimpleModel>>(x_data_, y_data_, 7, 2);

  optimizer.addCost(cost);

  for (int i = 0; i < 5; ++i) {
    optimizer.step(x0);
  }

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}
