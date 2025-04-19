#include "NumericalCost.hh"

static const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

/// \todo template float / double
NumericalCost::NumericalCost(const double* input, const double* observations, size_t num_elements, size_t output_dim,
                             size_t param_dim, const std::shared_ptr<IModel>& model, DifferentiationMethod method)
    : ICost(num_elements),
      input_(input),
      observations_(observations),
      output_dim_(output_dim),
      param_dim_(param_dim),
      model_(model),
      method_(method),
      residuals_dim_{num_elements * output_dim} {
  jacobian_data_.resize(residuals_dim_, param_dim_);
  residual_data_.resize(residuals_dim_);
  residual_data_plus_.resize(residuals_dim_);
  residual_data_minus_.resize(residuals_dim_);
}

/// \todo shared between analytical and numerical
/// \todo perhaps pass X dimensions at construction
/// \todo Eigen::Map?
double NumericalCost::computeCost(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  return residual_data_.squaredNorm();
}

ICost::SolveRhs NumericalCost::computeLinearSystem(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  // Compute residuals
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  // Compute differentials
  if (method_ == DifferentiationMethod::BACKWARD_EULER) {
    return applyEulerDiff(x);
  }

  return applyCentralDiff(x);
}

ICost::SolveRhs NumericalCost::applyEulerDiff(const Eigen::VectorXd& x) {
  for (int i = 0; i < param_dim_; ++i) {
    Eigen::VectorXd x_plus(x);
    x_plus[i] += g_step;

    model_->setup(x_plus.data());

    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_data_plus_[j]);
    }

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_) / g_step;
  }

  const auto&& JTJ = jacobian_data_.transpose() * jacobian_data_;
  const auto&& JTb = jacobian_data_.transpose() * residual_data_;
  return {JTJ, JTb, residual_data_.squaredNorm()};
}

ICost::SolveRhs NumericalCost::applyCentralDiff(const Eigen::VectorXd& x) {
  for (int i = 0; i < param_dim_; ++i) {
    Eigen::VectorXd x_plus(x);
    Eigen::VectorXd x_minus(x);
    x_plus[i] += g_step;
    x_minus[i] -= g_step;

    model_->setup(x_plus.data());
    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_data_plus_[j]);
    }

    model_->setup(x_minus.data());
    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_data_minus_[j]);
    }

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_minus_) / (2 * g_step);
  }
  const auto&& JTJ = jacobian_data_.transpose() * jacobian_data_;
  const auto&& JTb = jacobian_data_.transpose() * residual_data_;
  return {JTJ, JTb, residual_data_.squaredNorm()};
}