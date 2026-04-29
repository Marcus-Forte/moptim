#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
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
