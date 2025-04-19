#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;

/// \todo pipelines with differnet machines
TEST(TestJacobian, NumericalJacobianEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};
  NumericalCostSycl<SimpleModel> num_cost_sycl(queue, x_data_.data(), y_data_.data(), x_data_.size(), 1, 2);
  NumericalCost num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, std::make_shared<SimpleModel>());

  Eigen::VectorXd x{{0.1, 0.1}};

  const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = num_cost_sycl.computeLinearSystem(x);
  const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);

  std::cout << "num_jtj_sycl: " << num_jtj_sycl << std::endl;
  std::cout << "num_jtb_sycl: " << num_jtb_sycl << std::endl;
  std::cout << "num_total_sycl: " << num_total_sycl << std::endl;
  std::cout << "   \n";
  std::cout << "num_jtj: " << num_jtj << std::endl;
  std::cout << "num_jtb: " << num_jtb << std::endl;
  std::cout << "num_total: " << num_total << std::endl;
  EXPECT_NEAR(num_total_sycl, num_total, 1e-5);

  for (int i = 0; i < num_jtj_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtj_sycl(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < num_jtb_sycl.size(); ++i) {
    EXPECT_NEAR(num_jtb_sycl(i), num_jtb(i), 1e-5);
  }
}