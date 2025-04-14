#pragma once

#include <memory>

#include "ICost.hh"
#include "IModel.hh"

class NumericalCost : public ICost {
 public:
  NumericalCost(const NumericalCost&) = delete;

  NumericalCost(const double* input, const double* observations, size_t input_size, size_t output_dim,
                const std::shared_ptr<IModel>& model,
                DifferentiationMethod method = DifferentiationMethod::BACKWARD_EULER);

  ~NumericalCost() = default;

  double computeCost(const Eigen::VectorXd& x) override;

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override;

 private:
  SolveRhs applyEulerDiff(const Eigen::VectorXd& x);
  SolveRhs applyCentralDiff(const Eigen::VectorXd& x);

  const double* input_;
  const double* observations_;
  std::shared_ptr<IModel> model_;
  const size_t input_size_;
  const size_t output_dim_;
  const DifferentiationMethod method_;
};