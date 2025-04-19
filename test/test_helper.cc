
#include "test_helper.hh"

#include <gtest/gtest.h>

void compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance) {
  ASSERT_EQ(m1.size(), m2.size());

  for (int i = 0; i < m1.size(); ++i) {
    EXPECT_NEAR(m1(i), m2(i), tolerance);
  }
}