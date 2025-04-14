#include "NumericalCost2.hh"

static const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

NumericalCost2::NumericalCost2(const double* input, const double* observations, size_t input_size, size_t output_dim,
                               const std::shared_ptr<IModel>& model, DifferentiationMethod method)
    : input_(input),
      observations_(observations),
      input_size_(input_size),
      output_dim_(output_dim),
      model_(model),
      method_(method) {
        /// \todo preallocate matrices
        /// \todo template float / double
      }

double NumericalCost2::computeCost(const Eigen::VectorXd& x) {
  double sum = 0.0;

  model_->setup(x.data());

  Eigen::VectorXd residual(output_dim_);

  for (int i = 0; i < input_size_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], residual.data());
    sum += residual.squaredNorm();
  }

  return sum;
}

ICost::SolveRhs NumericalCost2::computeLinearSystem(const Eigen::VectorXd& x) {

  if (method_ == DifferentiationMethod::BACKWARD_EULER) {
    return applyEulerDiff(x);
  }

  return applyCentralDiff(x);
}

ICost::SolveRhs NumericalCost2::applyEulerDiff(const Eigen::VectorXd& x) {
  const auto param_dim = x.size();

  double sum = 0.0;

  Eigen::MatrixXd jacobian(output_dim_ * input_size_, param_dim);
  Eigen::VectorXd residual(output_dim_ * input_size_);
  Eigen::VectorXd residual_plus(output_dim_ * input_size_);

  model_->setup(x.data());

  // Compute Residuals
  for (int i = 0; i < input_size_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual[i]);
  }

  sum += residual.squaredNorm();

  // Compute differentials
  for (int i = 0; i < param_dim; ++i) {
    Eigen::VectorXd x_plus(x);
    x_plus[i] += g_step;
    model_->setup(x_plus.data());

    for (int j = 0; j < input_size_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_plus[j]);
    }

    jacobian.col(i) = (residual_plus - residual) / g_step;
  }
  const auto&& JTJ = jacobian.transpose() * jacobian;
  const auto&& JTb = jacobian.transpose() * residual;
  return {JTJ, JTb, sum};
}

ICost::SolveRhs NumericalCost2::applyCentralDiff(const Eigen::VectorXd& x) {
  const auto param_dim = x.size();

  double sum = 0.0;

  Eigen::MatrixXd jacobian(output_dim_ * input_size_, param_dim);
  Eigen::VectorXd residual(output_dim_ * input_size_);
  Eigen::VectorXd residual_plus(output_dim_ * input_size_);
  Eigen::VectorXd residual_minus(output_dim_ * input_size_);

  model_->setup(x.data());

  // Compute Residuals
  for (int i = 0; i < input_size_; i += output_dim_) {
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
    for (int j = 0; j < input_size_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_plus[j]);
    }

    model_->setup(x_minus.data());
    for (int j = 0; j < input_size_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_minus[j]);
    }

    jacobian.col(i) = (residual_plus - residual_minus) / (2 * g_step);
  }
  const auto&& JTJ = jacobian.transpose() * jacobian;
  const auto&& JTb = jacobian.transpose() * residual;
  return {JTJ, JTb, sum};
}