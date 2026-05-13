#pragma once

#include "moptim/IModel.h"
#include "gtest/gtest.h"

/**
 * @brief Common model to be used in the tests.
 *
 */

namespace test_models {

template <class T>
struct TestData {
  static constexpr int num_measurements = 7;
  static constexpr T x_data_[num_measurements]{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
  static constexpr T y_data_[num_measurements]{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
};

template <typename T>
class SimpleModelTest : public ::testing::Test {
 protected:
  TestData<T> test_data_;
};

template <class T>
struct SimpleModel {
  void setState(const T* /*x*/) {}

  void residual(const T* x, const T* input, const T* obs, T* res) {
    res[0] = obs[0] - x[0] * input[0] / (x[1] + input[0]);
  }

  void jacobian(const T* x, const T* input, const T* /*obs*/, T* jac) {
    const auto den = x[1] + input[0];
    jac[0] = -input[0] / den;
    jac[1] = x[0] * input[0] / (den * den);
  }
};

}  // namespace test_models