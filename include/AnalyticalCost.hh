#pragma once

#include <memory>

#include "ICost.hh"
#include "IModel.hh"

class AnalyticalCost : public ICost {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;

  AnalyticalCost(const double* input, const double* observations, size_t num_elements, size_t output_dim,
                 size_t param_dim, const std::shared_ptr<IJacobianModel>& model);

  double computeCost(const Eigen::VectorXd& x) override;

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override;

 private:
  Eigen::MatrixXd jacobian_transposed_data_;
  Eigen::VectorXd residual_data_;

  const double* input_;
  const double* observations_;
  std::shared_ptr<IJacobianModel> model_;
  const size_t output_dim_;
  const size_t param_dim_;
  const size_t residuals_dim_;
};