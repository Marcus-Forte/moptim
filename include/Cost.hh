#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#include "ICost.hh"
#include "ILog.hh"

template <class InputT, class OutputT, class Model>
class Cost : public ICost {
 public:
  Cost(const InputT* input, const OutputT* measurements, size_t num_elements, size_t param_dim)
      : input_(input), measurements_(measurements), num_elements_(num_elements), param_dim_(param_dim) {
    jacobian_.resize(num_elements_ * sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(num_elements_ * sizeof(OutputT) / sizeof(double));
  }

  ~Cost() = default;
  // {
  // delete[] error_plus_;
  // delete[] error_;
  // }

  double computeError(const Eigen::VectorXd& x) const override {
    return std::transform_reduce(
        input_, input_ + num_elements_, measurements_, 0.0, [](double a, double b) { return a * a + b * b; }, Model(x));
  }

  Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const override {
    auto* residual_ptr = reinterpret_cast<OutputT*>(residual_.data());
    std::transform(input_, input_ + num_elements_, measurements_, residual_ptr, Model(x));
    return residual_;
  }

  Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const override {
    Model model(x);
    const double step = std::sqrt(std::numeric_limits<double>::epsilon());

    for (size_t i = 0; i < param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);

      x_plus[i] += step;
      Model model_plus(x_plus);

      const auto model_diff = [&model, &model_plus, step](InputT input, OutputT measurement) {
        const auto err = (model_plus(input, measurement) - model(input, measurement)) / step;
        return err;
      };

      // Assume column major mem. layout.
      auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_.col(i).data());
      std::transform(input_, input_ + num_elements_, measurements_, jacobian_col, model_diff);
    }

    return jacobian_;
  }

 private:
  const InputT* input_;
  const OutputT* measurements_;
  const size_t num_elements_;
  const size_t param_dim_;

  mutable Eigen::MatrixXd jacobian_;
  mutable Eigen::VectorXd residual_;
};