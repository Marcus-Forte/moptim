#pragma once

#include <Eigen/Dense>

#include "IModel.hh"

/**
 * @brief Common model to be used in the tests.
 *
 */

namespace test_models {

extern const std::vector<double> x_data_;
extern const std::vector<double> y_data_;

struct SimpleModel : public IJacobianModel<double> {
  void setup(const double* x) final {
    x_[0] = x[0];
    x_[1] = x[1];
  }

  void f(const double* input, const double* measurement, double* f_x) final {
    f_x[0] = measurement[0] - x_[0] * input[0] / (x_[1] + input[0]);
  }

  void df(const double* input, const double* measurement, double* df_x) final {
    const auto den = (x_[1] + input[0]);
    df_x[0] = -input[0] / den;
    df_x[1] = x_[0] * input[0] / (den * den);
  }

  double x_[2];
};

}  // namespace test_models