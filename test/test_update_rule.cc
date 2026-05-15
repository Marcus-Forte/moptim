#include <gtest/gtest.h>

#include "ConsoleLogger.hh"
#include "moptim/GaussNewton.h"
#include "moptim/NumericalCostForwardEuler.h"

namespace {

struct DoubledStepPlusOperator {
  static void plus(const double* x, const double* delta, double* out, size_t dimensions) {
    for (size_t i = 0; i < dimensions; ++i) {
      out[i] = x[i] + 2.0 * delta[i];
    }
  }
};

struct OffsetLineModel {
  void setState(const double* x) { state_ = x[0]; }

  void residual(const double* /*x*/, const double* /*input*/, const double* obs, double* res) const {
    res[0] = obs[0] - state_;
  }

  double state_ = 0.0;
};

TEST(UpdateRule, NumericalForwardEulerUsesCustomPlus) {
  const double input[1] = {0.0};
  const double observation[1] = {1.0};

  moptim::NumericalCostForwardEuler<OffsetLineModel, double, DoubledStepPlusOperator> cost(input, observation, 1, 1, 1,
                                                                                           1);

  double x[1] = {0.0};
  double jtj[1] = {0.0};
  double jtb[1] = {0.0};
  double total = 0.0;

  cost.computeLinearSystem(x, jtj, jtb, total);

  EXPECT_NEAR(jtj[0], 4.0, 1e-5);
  EXPECT_NEAR(jtb[0], -2.0, 1e-5);
  EXPECT_NEAR(total, 1.0, 1e-8);
}

}  // namespace