#include "AnalyticalCost.hh"

AnalyticalCost::AnalyticalCost(const double* input, const double* observations, size_t num_elements, size_t output_dim,
                               size_t param_dim, const std::shared_ptr<IJacobianModel>& model)
    : ICost(num_elements),
      input_{input},
      observations_{observations},
      output_dim_{output_dim},
      param_dim_{param_dim},
      model_{model},
      residuals_dim_{num_elements * output_dim} {
  // We fill the jacobian transposed already
  jacobian_transposed_data_.resize(param_dim_, residuals_dim_);
  residual_data_.resize(residuals_dim_);
}

/// \todo shared between analytical and numerical
double AnalyticalCost::computeCost(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  return residual_data_.squaredNorm();
}

ICost::SolveRhs AnalyticalCost::computeLinearSystem(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  int k = 0;
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
    model_->df(&input_[i], &observations_[i], jacobian_transposed_data_.data() + k);
    k += param_dim_;
  }

  const auto&& JTJ = jacobian_transposed_data_ * jacobian_transposed_data_.transpose();
  const auto&& JTb = jacobian_transposed_data_ * residual_data_;
  return {JTJ, JTb, residual_data_.squaredNorm()};
}