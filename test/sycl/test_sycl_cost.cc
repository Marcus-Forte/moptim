#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "NumericalCost.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;

// apparently sycl can't handle inherited objects
struct SimpleModelSycl {
  void setup(const double* x) {
    x_[0] = x[0];
    x_[1] = x[1];
  }

  void f(const double* input, const double* measurement, double* f_x) {
    f_x[0] = measurement[0] - x_[0] * input[0] / (x_[1] + input[0]);
  }

  void df(const double* input, const double* measurement, double* df_x) {
    const auto den = (x_[1] + input[0]);
    df_x[0] = -input[0] / den;
    df_x[1] = x_[0] * input[0] / (den * den);
  }

  double x_[2];
};

/// \todo pipelines with differnet machines
TEST(TestCost, NumericalCostEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};

  const auto model = std::make_shared<SimpleModel>();

  NumericalCostSycl<SimpleModelSycl, 1> num_cost_sycl(queue, x_data_.data(), y_data_.data(), x_data_.size());
  NumericalCost num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, model);

  Eigen::VectorXd x{{0.0, 0.0}};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}

// TEST(TestCost, NumericalJacobianEquivalenceSycl) {
//   sycl::queue queue{sycl::default_selector_v};
//   NumericalCostSycl<double, double, SimpleModel, ::DifferentiationMethod::BACKWARD_EULER> num_cost_sycl(
//       &x_data_, &y_data_, queue);
//   NumericalCost<double, double, SimpleModel> num_cost(&x_data_, &y_data_);

//   Eigen::VectorXd x{{0.1, 0.1}};

//   const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = num_cost_sycl.computeLinearSystem(x);
//   const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);

//   EXPECT_NEAR(num_total_sycl, num_total, 1e-5);

//   for (int i = 0; i < num_jtj_sycl.size(); ++i) {
//     EXPECT_NEAR(num_jtj_sycl(i), num_jtj(i), 1e-5);
//   }

//   for (int i = 0; i < num_jtb_sycl.size(); ++i) {
//     EXPECT_NEAR(num_jtb_sycl(i), num_jtb(i), 1e-5);
//   }
// }