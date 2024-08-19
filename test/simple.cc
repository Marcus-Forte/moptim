#include <gtest/gtest.h>

#include <algorithm>

#include "Cost.hh"
#include "IOptmizer.hh"

double x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
double y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

void fun(const double* x, const double* input, double* output) { *output = x[0] * *(input) / x[1] + *input; }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

TEST(test_aa, bb) {
  double error[7];
  double xzero[2] = {0.1, 0.1};
  // use std::transform to calculare vector of errors defined by fun
}