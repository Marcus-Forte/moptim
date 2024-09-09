#include <gtest/gtest.h>

#include "GaussNewton.hh"
#include "NumericalCost.hh"
#include "models.hh"

class TestSimpleModel : public ::testing::Test {
 protected:
  const std::vector<double> x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
  const std::vector<double> y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement
};

TEST_F(TestSimpleModel, gauss_newton) {
  Eigen::VectorXd x{{0.9, 0.2}};
  GaussNewton solver;

  auto cost = std::make_shared<NumericalCost<double, double, SimpleModel, DifferentiationMethod::CENTRAL>>(&x_data_,
                                                                                                           &y_data_, 2);

  solver.addCost(cost);

  solver.optimize(x);

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}
