#pragma once

#include "BaseCost.hh"

template <class InputT, class OutputT, class Model>
class AnalyticalCost : public BaseCost<InputT, OutputT, Model> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  AnalyticalCost(size_t param_dim) : BaseCost<InputT, OutputT, Model>(param_dim) {}

  AnalyticalCost(const std::vector<InputT>* input, const std::vector<OutputT>* observations, size_t param_dim)
      : BaseCost<InputT, OutputT, Model>(input, observations, param_dim) {}

  const Eigen::MatrixXd& computeJacobian(const Eigen::VectorXd& x) override {
    using JacobianReturnType = typename std::result_of<decltype (&Model::jacobian)(Model, InputT, OutputT)>::type;
    Model model(x);
    const auto jacobian = [&model](InputT input, OutputT observation) -> JacobianReturnType {
      return model.jacobian(input, observation);
    };

    // // Create a lambda function to map to rowmajor
    auto* jacobian_row = reinterpret_cast<JacobianReturnType*>(this->jacobian_.data());
    std::transform(this->input_->begin(), this->input_->end(), this->observations_->begin(), jacobian_row, jacobian);
    return this->jacobian_;
  }
};