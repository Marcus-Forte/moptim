#pragma once

#include <execution>
#include <numeric>

#include "ICost.hh"
template <class InputT, class OutputT, class Model>
class AnalyticalCost : public ICost {
  static constexpr size_t OutputDim = sizeof(OutputT) / sizeof(double);
  using ResidualVectorT = Eigen::Vector<double, OutputDim>;

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
    Model model(x);

    const auto error_norm = [&model](const InputT& input, const OutputT& observation) -> double {
      const auto&& error = model(input, observation);
      Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&error));
      return residual_map.squaredNorm();
    };
    return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), 0.0,
                                 std::plus<>(), error_norm);
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override {
    using JacobianReturnType = typename std::result_of<decltype (&Model::jacobian)(Model, InputT, OutputT)>::type;

    SolveRhs init{Eigen::MatrixXd::Zero(x.size(), x.size()), Eigen::VectorXd::Zero(x.size()), 0.0};

    Model model(x);

    const auto&& jacobian = [&model, &x](InputT input, OutputT observation) -> SolveRhs {
      JacobianReturnType&& jacobian_matrix = model.jacobian(input, observation);
      OutputT&& residual = model(input, observation);

      Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
      const auto&& JTJ = jacobian_matrix.transpose() * jacobian_matrix;
      const auto&& JTb = jacobian_matrix.transpose() * residual_map;
      return {JTJ, JTb, residual_map.squaredNorm()};
    };

    return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
                                 ICost::Reduction, jacobian);
  }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  bool no_input_;
};