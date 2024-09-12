#pragma once

#include "ICost.hh"

template <class InputT, class OutputT, class Model>
class AnalyticalCost : public ICost {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  AnalyticalCost()
      : input_{new std::vector<InputT>{{}}}, observations_{new std::vector<OutputT>{{}}}, no_input_{true} {}
  AnalyticalCost(const std::vector<InputT>* input, const std::vector<OutputT>* observations)
      : input_{input}, observations_{observations}, no_input_{false} {}

  ~AnalyticalCost() {
    if (no_input_) {
      delete input_;
      delete observations_;
    }
  }

  double computeCost(const Eigen::VectorXd& x) override {
    residuals_.resize(input_->size() * sizeof(OutputT) / sizeof(double));
    Model model(x);
    std::transform(input_->begin(), input_->end(), observations_->begin(),
                   reinterpret_cast<OutputT*>(residuals_.data()), model);
    return residuals_.squaredNorm();
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override {
    using JacobianReturnType = typename std::result_of<decltype (&Model::jacobian)(Model, InputT, OutputT)>::type;
    jacobian_.resize(input_->size() * sizeof(OutputT) / sizeof(double), x.size());
    residuals_.resize(input_->size() * sizeof(OutputT) / sizeof(double));

    Model model(x);

    std::transform(input_->begin(), input_->end(), observations_->begin(),
                   reinterpret_cast<OutputT*>(residuals_.data()), model);

    const auto jacobian = [&model](InputT input, OutputT observation) -> JacobianReturnType {
      return model.jacobian(input, observation);
    };

    std::transform(input_->begin(), input_->end(), observations_->begin(),
                   reinterpret_cast<JacobianReturnType*>(jacobian_.data()), jacobian);

    // Reduce
    const auto JTJ = jacobian_.transpose() * jacobian_;
    const auto JTb = jacobian_.transpose() * residuals_;
    const auto totalCost = residuals_.squaredNorm();
    return {JTJ, JTb, totalCost};
  }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_;
  Eigen::VectorXd residuals_;
  bool no_input_;
};