#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

#include "ICost.hh"

const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

template <class InputT, class OutputT, class Model>
class Cost : public ICost {
 public:
  Cost(const InputT* input, const OutputT* measurements, size_t num_elements, size_t param_dim)
      : input_(input), measurements_(measurements), num_elements_(num_elements), param_dim_(param_dim) {
    jacobian_.resize(num_elements_ * sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(num_elements_ * sizeof(OutputT) / sizeof(double));
  }

  ~Cost() = default;

  // TODO
  double computeError(const Eigen::VectorXd& x) const override {
    return 0;
    // std::transform_reduce(
    //     input_, input_ + num_elements_, measurements_, 0.0,
    //     // Reduce
    //     [](OutputT a,  OutputT b)-> double { return 0; },
    //     // Transform
    //     Model(x));
  }

  Eigen::VectorXd computeResidual(const Eigen::VectorXd& x) const override {
    auto* residual_ptr = reinterpret_cast<OutputT*>(residual_.data());
    std::transform(input_, input_ + num_elements_, measurements_, residual_ptr, Model(x));
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
      std::transform(input_, input_ + num_elements_, measurements_, jacobian_col, model_diff);
    }

    return jacobian_;
  }

  // JTJ, JTb
  SolveRhs computeHessian(const Eigen::VectorXd& x) const override {
    const auto jacobian = computeJacobian(x);
    const auto residual = computeResidual(x);

    return {jacobian.transpose() * jacobian, jacobian.transpose() * residual};
  }

 private:
  const InputT* input_;
  const OutputT* measurements_;
  const size_t num_elements_;
  const size_t param_dim_;

  mutable Eigen::MatrixXd jacobian_;
  mutable Eigen::VectorXd residual_;
};