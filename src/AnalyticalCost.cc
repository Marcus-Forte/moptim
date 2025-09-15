#include "AnalyticalCost.hh"

namespace moptim {
template <class T>
AnalyticalCost<T>::AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t output_dim,
                                  size_t param_dim, const std::shared_ptr<IJacobianModel<T>>& model)
    : ICost<T>(param_dim, num_elements),
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
template <class T>
T AnalyticalCost<T>::computeCost(const T* x) {
  model_->setup(x);

  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
  }

  return residual_data_.squaredNorm();
}

template <class T>
void AnalyticalCost<T>::computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) {
  model_->setup(x);

  int k = 0;
  for (int i = 0; i < residuals_dim_; i += output_dim_) {
    model_->f(&input_[i], &observations_[i], &residual_data_[i]);
    model_->df(&input_[i], &observations_[i], jacobian_transposed_data_.data() + k);
    k += param_dim_;
  }

  Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_transposed_data_ * jacobian_transposed_data_.transpose();
  JTb_map = jacobian_transposed_data_ * residual_data_;
  *cost = residual_data_.squaredNorm();
}

template class AnalyticalCost<double>;
template class AnalyticalCost<float>;

}  // namespace moptim
