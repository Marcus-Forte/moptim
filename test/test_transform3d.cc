#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostCentral.hh"
#include "transform3d.hh"

using namespace moptim;

TEST_P(TestTransform3D, AnalyticalJacobianMatchesNumerical) {
  Eigen::VectorXd x = x0_ref;

  const size_t n = transformed_pointcloud_.size();
  constexpr size_t obs_dim = 3;
  constexpr size_t param_dim = 6;

  AnalyticalCost<Point3Distance, double> an_cost(transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, obs_dim,
                                                 obs_dim, param_dim);
  NumericalCostCentral<Point3Distance, double> num_cost(transformed_pointcloud_[0].data(), pointcloud_[0].data(), n,
                                                        obs_dim, obs_dim, param_dim);

  Eigen::MatrixXd an_jtj(param_dim, param_dim);
  Eigen::MatrixXd num_jtj(param_dim, param_dim);
  Eigen::VectorXd an_jtb(param_dim);
  Eigen::VectorXd num_jtb(param_dim);
  double an_total = 0.0;
  double num_total = 0.0;

  an_cost.computeLinearSystem(x.data(), an_jtj.data(), an_jtb.data(), an_total);
  num_cost.computeLinearSystem(x.data(), num_jtj.data(), num_jtb.data(), num_total);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(an_jtj(i), num_jtj(i), 1e-3) << "JTJ mismatch at index " << i;
  }
  for (int i = 0; i < an_jtb.size(); ++i) {
    EXPECT_NEAR(an_jtb(i), num_jtb(i), 1e-3) << "JTb mismatch at index " << i;
  }
  EXPECT_NEAR(an_total, num_total, 1e-10);
}

INSTANTIATE_TEST_SUITE_P(Transform3DJacobian, TestTransform3D, ::testing::Values(50, 200));

TEST_P(TestTransform3D, 3DTransformLMAnalytical) {
  auto logger = std::make_shared<ConsoleLogger>();
  auto solver = std::make_shared<LevenbergMarquardt<double>>(6, logger);

  const size_t n = transformed_pointcloud_.size();
  constexpr size_t obs_dim = 3;
  constexpr size_t param_dim = 6;

  auto cost = std::make_shared<AnalyticalCost<Point3Distance, double>>(
      transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, obs_dim, obs_dim, param_dim);
  solver->addCost(cost);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(param_dim);
  solver->optimize(x0.data());

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  EXPECT_NEAR(x0[3], -x0_ref[3], 1e-3);
  EXPECT_NEAR(x0[4], -x0_ref[4], 1e-3);
  EXPECT_NEAR(x0[5], -x0_ref[5], 1e-3);
}

INSTANTIATE_TEST_SUITE_P(Transform3DLM, TestTransform3D, ::testing::Values(50, 200));
