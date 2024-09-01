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

struct Point2 {
  double x;
  double y;
};

Point2 points[] = {{0, 0}, {1, 1}};
Point2 measured[] = {{0, 0}, {1, 1}};

TEST(test_simple, test_simple) { double x0[2] = {1.0, 1.0}; }