#pragma once

#include <Eigen/Dense>

struct SimpleModel {
  SimpleModel(const Eigen::VectorXd& x0) : x_(x0) {}

  double operator()(double input, double measurement) const { return measurement - x_[0] * input / (x_[1] + input); }
  Eigen::Vector2d jacobian(double input, double measurement) const {
    const auto den = (x_[1] + input);

    return {-input / den, x_[0] * input / (den * den)};
  }

  const Eigen::VectorXd x_;
};