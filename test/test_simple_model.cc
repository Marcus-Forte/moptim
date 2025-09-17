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

template <typename T>
class SimpleModelTest : public ::testing::Test {
 protected:
  const std::vector<T> x_data_;
  const std::vector<T> y_data_;

  SimpleModelTest()
      : x_data_{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70},
        y_data_{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317} {}
};

TYPED_TEST_SUITE(SimpleModelTest, TestTypes);

TYPED_TEST(SimpleModelTest, GaussNewton) {
  using T = TypeParam;
  Eigen::Vector<T, Eigen::Dynamic> x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<NumericalCostForwardEuler<T>>(this->x_data_.data(), this->y_data_.data(), 1, 1, 2,
                                                             this->x_data_.size(), model);

  GaussNewton<T> solver(2, std::make_shared<ConsoleLogger>(ILog::Level::DEBUG));

  solver.addCost(cost);

  auto status = solver.optimize(x.data());

  std::cout << "Optimization status: " << static_cast<int>(status) << std::endl;

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TYPED_TEST(SimpleModelTest, GaussNewtonAnalytical) {
  using T = TypeParam;
  Eigen::Vector<T, Eigen::Dynamic> x{{0.9, 0.2}};

  const auto model = std::make_shared<SimpleModel<T>>();
  auto cost = std::make_shared<AnalyticalCost<T>>(this->x_data_.data(), this->y_data_.data(), 1, 1, 2,
                                                  this->x_data_.size(), model);
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
  auto cost = std::make_shared<AnalyticalCost<T>>(this->x_data_.data(), this->y_data_.data(), 1, 1, 2,
                                                  this->x_data_.size(), model);

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
  auto cost = std::make_shared<AnalyticalCost<T>>(this->x_data_.data(), this->y_data_.data(), 1, 1, 2,
                                                  this->x_data_.size(), model);

  LevenbergMarquardt<T> solver(2, std::make_shared<ConsoleLogger>());

  solver.addCost(cost);

  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}