#pragma once

#include <execution>
#include <memory>
#include <numeric>

#include "ICost.hh"

const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };

template <class InputT, class OutputT, class Model,
          DifferentiationMethod MethodT = DifferentiationMethod::BACKWARD_EULER>
class NumericalCost : public ICost {
  static constexpr size_t OutputDim = sizeof(OutputT) / sizeof(double);
  using ResidualVectorT = Eigen::Vector<double, OutputDim>;

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
    residuals_.resize(input_->size() * OutputDim);
    Model model(x);
    std::transform(input_->begin(), input_->end(), observations_->begin(),
                   reinterpret_cast<OutputT*>(residuals_.data()), model);
    return residuals_.squaredNorm();
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override {
    SolveRhs init{Eigen::MatrixXd::Zero(x.size(), x.size()), Eigen::VectorXd::Zero(x.size()), 0.0};

    Model model(x);

    if constexpr (MethodT == DifferentiationMethod::BACKWARD_EULER) {
      return applyEulerDiff(x, model, init);
    } else {
      return applyCentralDiff(x, model, init);
    }
  }

 private:
  inline SolveRhs applyEulerDiff(const Eigen::VectorXd& x, Model& model, SolveRhs& init) {
    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(x.size());
    for (int i = 0; i < x.size(); ++i) {
      Eigen::VectorXd x_plus(x);
      x_plus[i] += g_step;
      models_plus[i] = std::make_shared<Model>(x_plus);
    }

    const auto jacobian = [&](InputT input, OutputT observation) -> SolveRhs {
      Eigen::MatrixXd JTJ(x.size(), x.size());
      Eigen::VectorXd JTb(x.size());
      Eigen::MatrixXd jacobian_matrix(OutputDim, x.size());

      const auto residual = model(input, observation);
      for (int i = 0; i < x.size(); ++i) {
        auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_matrix.col(i).data());
        *jacobian_col = ((*models_plus[i])(input, observation) - residual) / g_step;
      }

      Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
      JTJ = jacobian_matrix.transpose() * jacobian_matrix;
      JTb = jacobian_matrix.transpose() * residual_map;
      return {JTJ, JTb, residual_map.squaredNorm()};
    };

    return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
                                 ICost::Reduction, jacobian);
  }

  inline SolveRhs applyCentralDiff(const Eigen::VectorXd& x, Model& model, SolveRhs& init) {
    // Initialize vector of models
    std::vector<std::shared_ptr<Model>> models_plus(x.size());
    std::vector<std::shared_ptr<Model>> models_minus(x.size());

    for (int i = 0; i < x.size(); ++i) {
      Eigen::VectorXd x_plus(x);
      Eigen::VectorXd x_minus(x);
      x_plus[i] += g_step;
      x_minus[i] -= g_step;
      models_plus[i] = std::make_shared<Model>(x_plus);
      models_minus[i] = std::make_shared<Model>(x_minus);
    }

    const auto jacobian = [&](InputT input, OutputT observation) -> SolveRhs {
      Eigen::MatrixXd JTJ(x.size(), x.size());
      Eigen::VectorXd JTb(x.size());
      Eigen::MatrixXd jacobian_matrix(OutputDim, x.size());

      const auto residual = model(input, observation);
      for (int i = 0; i < x.size(); ++i) {
        auto* jacobian_col = reinterpret_cast<OutputT*>(jacobian_matrix.col(i).data());
        *jacobian_col = ((*models_plus[i])(input, observation) - (*models_minus[i])(input, observation)) / (2 * g_step);
      }

      Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&residual));
      JTJ = jacobian_matrix.transpose() * jacobian_matrix;
      JTb = jacobian_matrix.transpose() * residual_map;
      return {JTJ, JTb, residual_map.squaredNorm()};
    };

    return std::transform_reduce(std::execution::seq, input_->begin(), input_->end(), observations_->begin(), init,
                                 ICost::Reduction, jacobian);
  }

  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  Eigen::VectorXd residuals_;
  bool no_input_;
};