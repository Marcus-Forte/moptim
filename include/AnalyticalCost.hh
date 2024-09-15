#pragma once

#include <execution>
#include <numeric>

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
    using ResidualVectorT = Eigen::Vector<double, sizeof(OutputT) / sizeof(double)>;

    SolveRhs init{Eigen::MatrixXd::Zero(x.size(), x.size()), Eigen::VectorXd::Zero(x.size()), 0.0};

    Model model(x);

    const auto jacobian = [&model, &x](InputT input, OutputT observation) -> SolveRhs {
      Eigen::MatrixXd JTJ(x.size(), x.size());
      Eigen::VectorXd JTb(x.size());

      JacobianReturnType jacobian_matrix = model.jacobian(input, observation);
      OutputT residual = model(input, observation);

      Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
      JTJ = jacobian_matrix.transpose() * jacobian_matrix;
      JTb = jacobian_matrix.transpose() * residual_map;
      return {JTJ, JTb, residual_map.squaredNorm()};
    };

    const auto reduction = [](SolveRhs a, const SolveRhs& b) -> SolveRhs {
      auto& [JTJ_a, JTb_a, residual_a] = a;
      const auto& [JTJ_b, JTb_b, residual_b] = b;
      JTJ_a += JTJ_b;
      JTb_a += JTb_b;
      residual_a += residual_b;

      return a;
    };

    return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
                                 reduction, jacobian);
  }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  Eigen::VectorXd residuals_;
  bool no_input_;
};