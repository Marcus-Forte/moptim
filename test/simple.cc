#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>

#include "Cost.hh"
#include "IModel.hh"
#include "IOptmizer.hh"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

double x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};        // model
double y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};  // measurement

struct Model : public IModel {
  // State (x)
  Model(const double* x0) : IModel(x0) { std::cout << "setup once?\n"; }

  // Error function
  double operator()(const double input, const double measurement) const {
    return measurement - x_[0] * input / (x_[1] + input);
  }
};

TEST(test_simple, test_simple) {
  double x0[2] = {1.0, 1.0};
  Cost<double, double, Model> cost(x_data, y_data, 7, 2);

  auto err = cost.computeError(x0);
  // cost.computeJacobian(x0);
  std::cout << "error = " << err << std::endl;
}