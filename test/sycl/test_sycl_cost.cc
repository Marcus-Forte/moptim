#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

/// \todo pipelines with differnet machines
TEST(TestCost, NumericalCostEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};
  auto logger = std::make_shared<ConsoleLogger>();

  const auto model = std::make_shared<SimpleModel>();

  NumericalCostSycl<double, SimpleModel> num_cost_sycl(logger, queue, x_data_.data(), y_data_.data(), x_data_.size(), 1,
                                                       2);
  NumericalCostForwardEuler<double> num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, 2, model);

  Eigen::VectorXd x{{0.0, 0.0}};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}
