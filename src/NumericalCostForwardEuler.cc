#include "NumericalCostForwardEuler.hh"

#include <cmath>

namespace moptim {

template <class T>
NumericalCostForwardEuler<T>::NumericalCostForwardEuler(const T* input, const T* observations, size_t num_elements,
                                                        size_t output_dim, size_t param_dim,
                                                        const std::shared_ptr<IModel<T>>& model)
    : ICost<T>(param_dim, num_elements),
      input_(input),
      observations_(observations),
      output_dim_(output_dim),
      param_dim_(param_dim),
      model_(model),
      residuals_dim_{num_elements * output_dim} {
  jacobian_data_.resize(residuals_dim_, param_dim_);
  residual_data_.resize(residuals_dim_);
  residual_data_plus_.resize(residuals_dim_);
}

/// \todo shared between analytical and numerical
/// \todo Eigen::Map?
template <class T>
T NumericalCostForwardEuler<T>::computeCost(const T* x) {
  model_->setup(x);

  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  return residual_data_.squaredNorm();
}

template <class T>
void NumericalCostForwardEuler<T>::computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) {
  model_->setup(x);

  // Compute residuals
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  Eigen::Map<const VectorT> x_vec(x, param_dim_);

  const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());

  for (int i = 0; i < param_dim_; ++i) {
    VectorT x_plus(x_vec);
    x_plus[i] += g_step;

    model_->setup(x_plus.data());

    for (int j = 0; j < residuals_dim_; j += output_dim_) {
      model_->f(&input_[j], &observations_[j], &residual_data_plus_[j]);
    }

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_) / g_step;
  }

  Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_data_.transpose() * jacobian_data_;
  JTb_map = jacobian_data_.transpose() * residual_data_;
  *cost = residual_data_.squaredNorm();
}

template class NumericalCostForwardEuler<double>;
template class NumericalCostForwardEuler<float>;

}  // namespace moptim