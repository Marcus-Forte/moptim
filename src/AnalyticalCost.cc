#include "AnalyticalCost.hh"

/// \todo perhaps pass X dimensions at construction
AnalyticalCost::AnalyticalCost(const double* input, const double* observations, size_t input_size, size_t output_dim,
                               const std::shared_ptr<IJacobianModel>& model)
    : input_(input), observations_(observations), input_size_(input_size), output_dim_(output_dim), model_(model) {}

/// \todo shared between analytical and numerical
double AnalyticalCost::computeCost(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  Eigen::VectorXd residual(output_dim_ * input_size_);

  for (int i = 0; i < input_size_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
  }

  return residual.squaredNorm();
}

ICost::SolveRhs AnalyticalCost::computeLinearSystem(const Eigen::VectorXd& x) {
  const auto param_dim = x.size();

  model_->setup(x.data());

  Eigen::MatrixXd jacobian_transposed(param_dim, output_dim_ * input_size_);  // this is the transpose
  Eigen::VectorXd residual(output_dim_ * input_size_);

  int k = 0;
  for (int i = 0; i < input_size_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
    model_->df(&input_[i], &observations_[i], jacobian_transposed.data() + k);
    k += param_dim;
  }

  const auto&& JTJ = jacobian_transposed * jacobian_transposed.transpose();
  const auto&& JTb = jacobian_transposed * residual;
  return {JTJ, JTb, residual.squaredNorm()};
}