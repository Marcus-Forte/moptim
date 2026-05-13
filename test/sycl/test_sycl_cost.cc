#include <gtest/gtest.h>

#include "moptim/AnalyticalCost.h"
#include "ConsoleLogger.hh"
#include "moptim/NumericalCostForwardEuler.h"
#include "moptim/NumericalCostSycl.h"
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

  NumericalCostSycl<T, SimpleModel<T>> num_cost_sycl(
      logger, queue, std::span<const T>(this->test_data_.x_data_, this->test_data_.num_measurements),
      std::span<const T>(this->test_data_.y_data_, this->test_data_.num_measurements), 1, 1, 2,
      this->test_data_.num_measurements);

  NumericalCostForwardEuler<SimpleModel<T>, T> num_cost(this->test_data_.x_data_, this->test_data_.y_data_,
                                                        this->test_data_.num_measurements, 1, 1, 2);

  T x[2]{0.0, 0.0};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cpu_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cpu_cost_result, 1e-5);
}
