#pragma once

#include <memory>

#include "ICost.hh"
#include "IModel.hh"

class NumericalCost : public ICost {
 public:
  NumericalCost(const NumericalCost&) = delete;

  NumericalCost(const double* input, const double* observations, size_t num_elements, size_t output_dim,
                size_t param_dim, const std::shared_ptr<IModel>& model,
                DifferentiationMethod method = DifferentiationMethod::BACKWARD_EULER);

  virtual ~NumericalCost() = default;

  double computeCost(const Eigen::VectorXd& x) override;

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override;

 private:
  SolveRhs applyEulerDiff(const Eigen::VectorXd& x);
  SolveRhs applyCentralDiff(const Eigen::VectorXd& x);

  Eigen::MatrixXd jacobian_data_;
  Eigen::VectorXd residual_data_;
  Eigen::VectorXd residual_data_plus_;
  Eigen::VectorXd residual_data_minus_;

  using ICost::num_elements_;

  const double* input_;
  const double* observations_;
  const size_t output_dim_;
  const size_t param_dim_;
  const size_t residuals_dim_;
  std::shared_ptr<IModel> model_;
  const DifferentiationMethod method_;
};