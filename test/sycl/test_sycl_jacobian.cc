#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "NumericalCostSycl.hh"
#include "test_helper.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

/// \todo pipelines with differnet machines
TEST(TestJacobian, NumericalJacobianEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  auto logger = std::make_shared<ConsoleLogger>();

  const auto* x_data_ = TestData<double>::x_data_;
  const auto* y_data_ = TestData<double>::y_data_;
  const auto num_measurements = TestData<double>::num_measurements;

  NumericalCostSycl<double, SimpleModel<double>> num_cost_sycl(logger, queue, x_data_, y_data_, 1, 1, 2,
                                                               num_measurements);

  NumericalCostForwardEuler<double> num_cost(x_data_, y_data_, 1, 1, 2, num_measurements,
                                             std::make_shared<SimpleModel<double>>());

  double x[2]{0.1, 0.1};

  Eigen::Matrix<double, 2, 2> jtj_sycl;
  Eigen::Matrix<double, 2, 1> jtb_sycl;
  double total_sycl = 0.0;

  Eigen::Matrix<double, 2, 2> jtj;
  Eigen::Matrix<double, 2, 1> jtb;
  double total = 0.0;

  num_cost_sycl.computeLinearSystem(x, jtj_sycl.data(), jtb_sycl.data(), &total_sycl);
  num_cost.computeLinearSystem(x, jtj.data(), jtb.data(), &total);

  std::cout << "num_jtj_sycl: " << jtj_sycl << std::endl;
  std::cout << "num_jtb_sycl: " << jtb_sycl << std::endl;
  std::cout << "num_total_sycl: " << total_sycl << std::endl;
  std::cout << "   \n";
  std::cout << "num_jtj: " << jtj << std::endl;
  std::cout << "num_jtb: " << jtb << std::endl;
  std::cout << "num_total: " << total << std::endl;

  EXPECT_NEAR(total_sycl, total, 1e-5);

  compareMatrices(jtj_sycl, jtj);
  compareMatrices(jtb_sycl, jtb);
}

// TEST(TestJacobian, NumericalJacobianMethods) {
//   sycl::queue queue{sycl::default_selector_v};
//   auto logger = std::make_shared<ConsoleLogger>();
//   const auto num_elements = x_data_.size();
//   NumericalCostSycl<SimpleModel> num_euler(logger, queue, x_data_.data(), y_data_.data(), num_elements, 1, 2);
//   NumericalCostSycl<SimpleModel> num_central(logger, queue, x_data_.data(), y_data_.data(), num_elements, 1, 2,
//                                              DifferentiationMethod::CENTRAL);

//   Eigen::VectorXd x{{0.1, 0.1}};

//   const auto [num_jtj_sycl, num_jtb_sycl, num_total_sycl] = num_euler.computeLinearSystem(x);
//   const auto [num_jtj, num_jtb, num_total] = num_central.computeLinearSystem(x);

//   EXPECT_NEAR(num_total_sycl, num_total, 1e-5);

//   compareMatrices(num_jtj_sycl, num_jtj);
//   compareMatrices(num_jtb_sycl, num_jtb);
// }