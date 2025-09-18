#pragma once

#include <Eigen/Dense>

#include "IModel.hh"

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

template <class T>
struct SimpleModel : public IJacobianModel<T> {
  void setup(const T* x) final {
    x_[0] = x[0];
    x_[1] = x[1];
  }

  void f(const T* input, const T* measurement, T* f_x) final {
    f_x[0] = measurement[0] - x_[0] * input[0] / (x_[1] + input[0]);
  }

  void df(const T* input, const T* measurement, T* df_x) final {
    const auto den = (x_[1] + input[0]);
    df_x[0] = -input[0] / den;
    df_x[1] = x_[0] * input[0] / (den * den);
  }

  T x_[2];
};

}  // namespace test_models