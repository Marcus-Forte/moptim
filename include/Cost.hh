#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

#include "ICost.hh"

const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

template <class InputT, class OutputT, class Model>
class Cost : public ICost {
 public:
  Cost(const Cost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  Cost(size_t param_dim)
      : param_dim_(param_dim),
        input_{new std::vector<InputT>{{}}},
        measurements_{new std::vector<OutputT>{{}}},
        no_input_(true) {
    jacobian_.resize(sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(sizeof(OutputT) / sizeof(double));
  }
  Cost(const std::vector<InputT>* input, const std::vector<OutputT>* measurements, size_t param_dim)
      : input_(input), measurements_(measurements), param_dim_(param_dim), no_input_(false) {
    jacobian_.resize(input_->size() * sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(input_->size() * sizeof(OutputT) / sizeof(double));
  }

  ~Cost() {
    if (no_input_) {
      delete input_;
      delete measurements_;
    }
  }

  double getCost(const Eigen::VectorXd& x) const override {
    // Sum squared errors
    return std::reduce(residual_.begin(), residual_.end(), 0.0, [](double a, double b) { return a * a + b * b; });
  }

  Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const override {
    auto* residual_ptr = reinterpret_cast<OutputT*>(residual_.data());
    std::transform(input_->begin(), input_->end(), measurements_->begin(), residual_ptr, Model(x));
    return residual_;
  }

  // Note: OutputT operator- and operator/ must be implemented.
  Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const override {
    Model model(x);

    for (size_t i = 0; i < param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_step;
      Model model_plus(x_plus);

      const auto model_diff = [&](InputT input, OutputT measurement) -> OutputT {
        return (model_plus(input, measurement) - model(input, measurement)) / g_step;
      };

      // Assume column major mem. layout.
      auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_.col(i).data());
      std::transform(input_->begin(), input_->end(), measurements_->begin(), jacobian_col, model_diff);
    }

    return jacobian_;
  }

  // TODO covariance
  SolveRhs computeHessian(const Eigen::VectorXd& x) const override {
    const auto jacobian = computeJacobian(x);
    const auto residual = computeResidual(x);
    const auto cost = getCost(x);
    return {jacobian.transpose() * jacobian, jacobian.transpose() * residual, cost};
  }

 private:
  const std::vector<InputT>* input_;
  const std::vector<OutputT>* measurements_;
  const size_t param_dim_;

  mutable Eigen::MatrixXd jacobian_;
  mutable Eigen::VectorXd residual_;

  bool no_input_;
};