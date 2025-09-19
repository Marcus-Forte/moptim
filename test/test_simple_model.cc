#include <gtest/gtest.h>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostForwardEuler.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

using TestTypes = ::testing::Types<double, float>;

TYPED_TEST_SUITE(SimpleModelTest, TestTypes);

TYPED_TEST(SimpleModelTest, GaussNewton) {
  using T = TypeParam;

  T x[]{0.9, 0.2};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<NumericalCostForwardEuler<T>>(this->test_data_.x_data_, this->test_data_.y_data_, 1, 1,
                                                             2, this->test_data_.num_measurements, model);

  GaussNewton<T> solver(2, std::make_shared<ConsoleLogger>(ILog::Level::DEBUG));

  solver.addCost(cost);

  auto status = solver.optimize(x);

  std::cout << "Optimization status: " << static_cast<int>(status) << std::endl;

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TYPED_TEST(SimpleModelTest, GaussNewtonAnalytical) {
  using T = TypeParam;
  Eigen::Vector<T, Eigen::Dynamic> x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<AnalyticalCost<T>>(this->test_data_.x_data_, this->test_data_.y_data_, 1, 1, 2,
                                                  this->test_data_.num_measurements, model);
  GaussNewton<T> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TYPED_TEST(SimpleModelTest, LevenbergMarquardt) {
  using T = TypeParam;
  Eigen::Vector<T, Eigen::Dynamic> x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<AnalyticalCost<T>>(this->test_data_.x_data_, this->test_data_.y_data_, 1, 1, 2,
                                                  this->test_data_.num_measurements, model);

  LevenbergMarquardt<T> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TYPED_TEST(SimpleModelTest, LevenbergMarquardtAnalytical) {
  using T = TypeParam;
  Eigen::Vector<T, Eigen::Dynamic> x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<AnalyticalCost<T>>(this->test_data_.x_data_, this->test_data_.y_data_, 1, 1, 2,
                                                  this->test_data_.num_measurements, model);

  LevenbergMarquardt<T> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}