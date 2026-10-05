#include <gtest/gtest.h>

#include "moptim/LossFunction.hh"

using namespace moptim;

TEST(LossFunction, TrivialLossIsIdentity) {
  TrivialLoss<double> loss;
  double out[3];
  loss.evaluate(4.0, out);
  EXPECT_DOUBLE_EQ(out[0], 4.0);
  EXPECT_DOUBLE_EQ(out[1], 1.0);
  EXPECT_DOUBLE_EQ(out[2], 0.0);
}

TEST(LossFunction, HuberInliersMatchTrivial) {
  HuberLoss<double> loss(1.0);
  double out[3];
  loss.evaluate(0.25, out);  // s <= delta^2
  EXPECT_DOUBLE_EQ(out[0], 0.25);
  EXPECT_DOUBLE_EQ(out[1], 1.0);
  EXPECT_DOUBLE_EQ(out[2], 0.0);
}

TEST(LossFunction, HuberOutliersAreSublinear) {
  HuberLoss<double> loss(1.0);
  double out[3];
  loss.evaluate(100.0, out);  // s >> delta^2
  EXPECT_DOUBLE_EQ(out[0], 2.0 * 1.0 * std::sqrt(100.0) - 1.0);
  EXPECT_DOUBLE_EQ(out[1], 1.0 / std::sqrt(100.0));
  EXPECT_LT(out[0], 100.0);  // grows much slower than the quadratic
}

TEST(LossFunction, HuberSetScale) {
  HuberLoss<double> loss(1.0);
  loss.setScale(2.0);
  EXPECT_DOUBLE_EQ(loss.scale(), 2.0);
  double out[3];
  loss.evaluate(1.0, out);  // s=1 <= delta^2=4 -> inlier regime now
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 1.0);
}

TEST(LossFunction, CauchyMatchesClosedForm) {
  CauchyLoss<double> loss(2.0);
  double out[3];
  const double s = 3.0;
  const double c2 = 4.0;
  loss.evaluate(s, out);
  EXPECT_NEAR(out[0], c2 * std::log(1.0 + s / c2), 1e-12);
  EXPECT_NEAR(out[1], c2 / (c2 + s), 1e-12);
  EXPECT_NEAR(out[2], -c2 / ((c2 + s) * (c2 + s)), 1e-12);
}

TEST(LossFunction, GemanMcClureMatchesClosedForm) {
  GemanMcClureLoss<double> loss(1.5);
  double out[3];
  const double s = 2.0;
  const double kappa = 1.5;
  loss.evaluate(s, out);
  EXPECT_NEAR(out[0], 0.5 * s / (kappa + s), 1e-12);
  EXPECT_NEAR(out[1], 0.5 * kappa / ((kappa + s) * (kappa + s)), 1e-12);
  EXPECT_NEAR(out[2], -kappa / ((kappa + s) * (kappa + s) * (kappa + s)), 1e-12);
}

TEST(LossFunction, LossesDownweightLargeResiduals) {
  // rho'(s) should shrink monotonically toward 0 as s grows, for every
  // non-trivial kernel. Huber and Cauchy additionally start at rho'(0) = 1.
  HuberLoss<double> huber(1.0);
  CauchyLoss<double> cauchy(1.0);

  for (auto* loss : {static_cast<LossFunction<double>*>(&huber), static_cast<LossFunction<double>*>(&cauchy)}) {
    double small[3];
    double large[3];
    loss->evaluate(1e-6, small);
    loss->evaluate(1e6, large);
    EXPECT_NEAR(small[1], 1.0, 1e-3);
    EXPECT_LT(large[1], 2e-3);
  }

  GemanMcClureLoss<double> gmc(1.0);
  double gmc_small[3];
  double gmc_mid[3];
  double gmc_large[3];
  gmc.evaluate(1e-6, gmc_small);
  gmc.evaluate(1.0, gmc_mid);
  gmc.evaluate(1e6, gmc_large);
  EXPECT_GT(gmc_small[1], gmc_mid[1]);
  EXPECT_GT(gmc_mid[1], gmc_large[1]);
  EXPECT_LT(gmc_large[1], 1e-6);
}
