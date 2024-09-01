#include <gtest/gtest.h>

#include <algorithm>
#include <format>
#include <memory>
#include <numeric>

#include "Cost.hh"
#include "IModel.hh"

const double x_data_[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
const double y_data_[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

struct Model : public IModel {
  // State (x)
  Model(const Eigen::VectorXd& x0) : IModel(x0) {}

  // Error function
  double operator()(double input, double measurement) const { return measurement - (x_[0] * input) / (x_[1] + input); }
};

TEST(test_simple, test_simple) {
  Eigen::VectorXd x0{{0.9, 0.2}};

  Cost<double, double, Model> cost(x_data_, y_data_, 7, 2);

  for (int i = 0; i < 5; ++i) {
    const auto jacobian = cost.computeJacobian(x0);

    const auto residual = cost.computeResidual(x0);

    const auto jtj = jacobian.transpose() * jacobian;
    auto Jtb = jacobian.transpose() * residual;

    // std::cout << "Hessian:  " << jtj << std::endl;
    // std::cout << "Res:  " << residual << std::endl;

    Eigen::LDLT<Eigen::Matrix<double, 2, 2>> solver(jtj);
    const auto x_plus = solver.solve(-Jtb);

    x0 += x_plus;

    // std::cout << "x0 = " << x0 << std::endl;
  }

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}