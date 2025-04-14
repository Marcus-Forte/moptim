#include <gtest/gtest.h>

#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;

/// \todo pipelines with differnet machines
TEST(TestCost, NumericalCostEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};
  NumericalCostSycl<double, double, SimpleModel> num_cost_sycl(&x_data_, &y_data_, queue);
  NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_);

  Eigen::VectorXd x{{0.1, 0.1}};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}

TEST(TestCost, NumericalJacobianEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};
  NumericalCostSycl<double, double, SimpleModel, ::DifferentiationMethod::BACKWARD_EULER> num_cost_sycl(
      &x_data_, &y_data_, queue);
  NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_);

  Eigen::VectorXd x{{0.1, 0.1}};

  const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = num_cost_sycl.computeLinearSystem(x);
  const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);

  EXPECT_NEAR(num_total_sycl, num_total, 1e-5);

  for (int i = 0; i < num_jtj_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtj_sycl(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < num_jtb_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtb_sycl(i), num_jtb(i), 1e-5);
  }
}