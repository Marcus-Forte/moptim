#pragma once

#include "ICost.hh"

const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

template <class InputT, class OutputT, class Model,
          DifferentiationMethod MethodT = DifferentiationMethod::BACKWARD_EULER>
class NumericalCost : public ICost {
 public:
  NumericalCost(const NumericalCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  NumericalCost() : input_{new std::vector<InputT>{{}}}, observations_{new std::vector<OutputT>{{}}}, no_input_(true) {}
  NumericalCost(const std::vector<InputT>* input, const std::vector<OutputT>* observations)
      : input_{input}, observations_{observations}, no_input_(false) {}

  ~NumericalCost() {
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
    jacobian_.resize(input_->size() * sizeof(OutputT) / sizeof(double), x.size());
    residuals_.resize(input_->size() * sizeof(OutputT) / sizeof(double));

    Model model(x);

    std::transform(input_->begin(), input_->end(), observations_->begin(),
                   reinterpret_cast<OutputT*>(residuals_.data()), model);

    if constexpr (MethodT == DifferentiationMethod::BACKWARD_EULER) {
      applyEulerDiff(x);
    } else {
      applyCentralDiff(x);
    }

    // Reduce
    const auto JTJ = jacobian_.transpose() * jacobian_;
    const auto JTb = jacobian_.transpose() * residuals_;
    const auto totalCost = residuals_.squaredNorm();
    return {JTJ, JTb, totalCost};
  }

 private:
  inline void applyEulerDiff(const Eigen::VectorXd& x) {
    Model model(x);
    for (size_t i = 0; i < x.size(); ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_step;
      Model model_plus(x_plus);

      const auto model_diff = [&](InputT input, OutputT measurement) -> OutputT {
        return (model_plus(input, measurement) - model(input, measurement)) / g_step;
      };

      // Assume column major mem. layout.
      std::transform(input_->begin(), input_->end(), observations_->begin(),
                     reinterpret_cast<OutputT*>(this->jacobian_.col(i).data()), model_diff);
    }
  }

  inline void applyCentralDiff(const Eigen::VectorXd& x) {
    Model model(x);
    for (size_t i = 0; i < x.size(); ++i) {
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
      std::transform(input_->begin(), input_->end(), observations_->begin(),
                     reinterpret_cast<OutputT*>(this->jacobian_.col(i).data()), model_diff);
    }
  }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd residuals_;
  bool no_input_;
};