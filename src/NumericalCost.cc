#include "NumericalCost.hh"

static const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

/// \todo perhaps pass X dimensions at construction
NumericalCost::NumericalCost(const double* input, const double* observations, size_t input_size, size_t output_dim,
                             const std::shared_ptr<IModel>& model, DifferentiationMethod method)
    : input_(input),
      observations_(observations),
      output_dim_(output_dim),
      model_(model),
      method_(method),
      residuals_dim_{input_size * output_dim} {
  /// \todo preallocate matrices
  /// \todo template float / double
}

/// \todo shared between analytical and numerical
/// \todo perhaps pass X dimensions at construction
/// \todo Eigen::Map?
double NumericalCost::computeCost(const Eigen::VectorXd& x) {
  model_->setup(x.data());

  Eigen::VectorXd residual(residuals_dim_);

  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
  }

  return residual.squaredNorm();
}

ICost::SolveRhs NumericalCost::computeLinearSystem(const Eigen::VectorXd& x) {
  if (method_ == DifferentiationMethod::BACKWARD_EULER) {
    return applyEulerDiff(x);
  }

  return applyCentralDiff(x);
}

ICost::SolveRhs NumericalCost::applyEulerDiff(const Eigen::VectorXd& x) {
  const auto param_dim = x.size();

  double sum = 0.0;

  Eigen::MatrixXd jacobian(residuals_dim_, param_dim);
  Eigen::VectorXd residual(residuals_dim_);
  Eigen::VectorXd residual_plus(residuals_dim_);

  model_->setup(x.data());

  // Compute Residuals
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
  }

  // Compute differentials
  for (int i = 0; i < param_dim; ++i) {
    Eigen::VectorXd x_plus(x);
    x_plus[i] += g_step;
    model_->setup(x_plus.data());

    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_plus[j]);
    }

    jacobian.col(i) = (residual_plus - residual) / g_step;
  }

  const auto&& JTJ = jacobian.transpose() * jacobian;
  const auto&& JTb = jacobian.transpose() * residual;
  return {JTJ, JTb, residual.squaredNorm()};
}

ICost::SolveRhs NumericalCost::applyCentralDiff(const Eigen::VectorXd& x) {
  const auto param_dim = x.size();

  double sum = 0.0;

  Eigen::MatrixXd jacobian(residuals_dim_, param_dim);
  Eigen::VectorXd residual(residuals_dim_);
  Eigen::VectorXd residual_plus(residuals_dim_);
  Eigen::VectorXd residual_minus(residuals_dim_);

  model_->setup(x.data());

  // Compute Residuals
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
  }

  sum += residual.squaredNorm();

  // Compute differentials
  for (int i = 0; i < param_dim; ++i) {
    Eigen::VectorXd x_plus(x);
    Eigen::VectorXd x_minus(x);
    x_plus[i] += g_step;
    x_minus[i] -= g_step;

    model_->setup(x_plus.data());
    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_plus[j]);
    }

    model_->setup(x_minus.data());
    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_minus[j]);
    }

    jacobian.col(i) = (residual_plus - residual_minus) / (2 * g_step);
  }
  const auto&& JTJ = jacobian.transpose() * jacobian;
  const auto&& JTb = jacobian.transpose() * residual;
  return {JTJ, JTb, sum};
}