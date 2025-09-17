#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

const double x_data_[]{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
const double y_data_[]{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

/// \todo pipelines with differnet machines
TEST(TestCost, NumericalCostEquivalenceSycl) {
  sycl::queue queue{sycl::default_selector_v};
  auto logger = std::make_shared<ConsoleLogger>();

  const auto model = std::make_shared<SimpleModel<double>>();

  NumericalCostSycl<double, SimpleModel<double>> num_cost_sycl(logger, queue, x_data_, y_data_, 1, 1, 2, 7);
  NumericalCostForwardEuler<double> num_cost(x_data_, y_data_, 1, 1, 2, 7, model);

  double x[2]{0.0, 0.0};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}
