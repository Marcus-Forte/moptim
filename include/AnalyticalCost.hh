#pragma once

#include <iostream>

#include "BaseCost.hh"
template <class InputT, class OutputT, class Model>
class AnalyticalCost : public BaseCost<InputT, OutputT, Model> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  AnalyticalCost(size_t param_dim) : AnalyticalCost<InputT, OutputT, Model>(param_dim) {}

  AnalyticalCost(const std::vector<InputT>* input, const std::vector<OutputT>* measurements, size_t param_dim)
      : BaseCost<InputT, OutputT, Model>(input, measurements, param_dim) {}

  Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& x) const override {
    Model model(x);

    const auto jacobian = [&](InputT input, OutputT measurement) { return model.jacobian(input, measurement); };

    // Assume column major mem. layout.
    // auto* jacobian_col = reinterpret_cast<OutputT*>(this->jacobian_.rowwise());
    // for (auto& row : this->jacobian_.rowwise()) {
    //   row = {1, 1};
    // }
    // std::transform(this->input_->begin(), this->input_->end(), this->measurements_->begin(),
    // this->jacobian_.rowwise(),
    //                jacobian);
    return this->jacobian_;
  }
};