#pragma once

#include "BaseCost.hh"

template <class InputT, class OutputT, class Model>
class NumericalCost : public BaseCost<InputT, OutputT, Model> {
 public:
  NumericalCost(const NumericalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  NumericalCost(size_t param_dim) : BaseCost<InputT, OutputT, Model>(param_dim) {}

  NumericalCost(const std::vector<InputT>* input, const std::vector<OutputT>* measurements, size_t param_dim)
      : BaseCost<InputT, OutputT, Model>(input, measurements, param_dim) {}

  Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) override {
    Model model(x);

    for (size_t i = 0; i < this->param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_step;
      Model model_plus(x_plus);

      const auto model_diff = [&model, &model_plus](InputT input, OutputT measurement) -> OutputT {
        return (model_plus(input, measurement) - model(input, measurement)) / g_step;
      };

      // Assume column major mem. layout.
      auto* jacobian_col = reinterpret_cast<OutputT*>(this->jacobian_.col(i).data());
      std::transform(this->input_->begin(), this->input_->end(), this->measurements_->begin(), jacobian_col,
                     model_diff);
    }

    return this->jacobian_;
  }
};