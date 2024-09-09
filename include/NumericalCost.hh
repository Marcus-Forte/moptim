#pragma once

#include "BaseCost.hh"

const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

template <class InputT, class OutputT, class Model,
          DifferentiationMethod MethodT = DifferentiationMethod::BACKWARD_EULER>
class NumericalCost : public BaseCost<InputT, OutputT, Model> {
 public:
  NumericalCost(const NumericalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  NumericalCost(size_t param_dim) : BaseCost<InputT, OutputT, Model>(param_dim) {}

  NumericalCost(const std::vector<InputT>* input, const std::vector<OutputT>* observations, size_t param_dim)
      : BaseCost<InputT, OutputT, Model>(input, observations, param_dim) {}

  Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) override {
    if constexpr (MethodT == DifferentiationMethod::BACKWARD_EULER) {
      EulerDiff(x);
    } else {
      CentralDiff(x);
    }

    return this->jacobian_;
  }

 private:
  inline void CentralDiff(const Eigen::VectorXd& x) {
    Model model(x);
    for (size_t i = 0; i < this->param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      Eigen::VectorXd x_minus(x);
      x_plus[i] += g_step;
      x_minus[i] -= g_step;

      Model model_plus(x_plus);
      Model model_minus(x_minus);

      const auto model_diff = [&](InputT input, OutputT measurement) -> OutputT {
        return (model_plus(input, measurement) - model_minus(input, measurement)) / (2 * g_step);
      };

      // Assume column major mem. layout.
      auto* jacobian_col = reinterpret_cast<OutputT*>(this->jacobian_.col(i).data());
      std::transform(this->input_->begin(), this->input_->end(), this->observations_->begin(), jacobian_col,
                     model_diff);
    }
  }
  inline void EulerDiff(const Eigen::VectorXd& x) {
    Model model(x);

    for (size_t i = 0; i < this->param_dim_; ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_step;
      Model model_plus(x_plus);

      const auto model_diff = [&](InputT input, OutputT measurement) -> OutputT {
        return (model_plus(input, measurement) - model(input, measurement)) / g_step;
      };

      // Assume column major mem. layout.
      auto* jacobian_col = reinterpret_cast<OutputT*>(this->jacobian_.col(i).data());
      std::transform(this->input_->begin(), this->input_->end(), this->observations_->begin(), jacobian_col,
                     model_diff);
    }
  }
};