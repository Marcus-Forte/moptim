#pragma once

#include <Eigen/Dense>

#include "IModel.hh"

/**
 * @brief Common model to be used in the tests.
 *
 */

namespace test_models {

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