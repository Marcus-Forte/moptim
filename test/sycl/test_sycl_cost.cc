#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCostForwardEuler.hh"
#include "NumericalCostSycl.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

using TestTypes = ::testing::Types<double, float>;

TYPED_TEST_SUITE(SimpleModelTest, TestTypes);

/// \todo pipelines with differnet machines
TYPED_TEST(SimpleModelTest, NumericalCostEquivalenceSycl) {
  using T = TypeParam;

  sycl::queue queue{sycl::default_selector_v};
  auto logger = std::make_shared<ConsoleLogger>();

  const auto model = std::make_shared<SimpleModel<T>>();

  NumericalCostSycl<T, SimpleModel<T>> num_cost_sycl(logger, queue, this->test_data_.x_data_, this->test_data_.y_data_,
                                                     1, 1, 2, this->test_data_.num_measurements);

  NumericalCostForwardEuler<T> num_cost(this->test_data_.x_data_, this->test_data_.y_data_, 1, 1, 2,
                                        this->test_data_.num_measurements, model);

  T x[2]{0.0, 0.0};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}
