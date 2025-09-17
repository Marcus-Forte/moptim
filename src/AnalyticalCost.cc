#include "AnalyticalCost.hh"

namespace moptim {
template <class T>
AnalyticalCost<T>::AnalyticalCost(const T* input, const T* observations, size_t input_dim, size_t observation_dim,
                                  size_t param_dim, size_t num_elements,
                                  const std::shared_ptr<IJacobianModel<T>>& model)
    : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
      input_{input},
      observations_{observations},
      model_{model} {
  // We fill the jacobian transposed already
  jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements);
  residual_data_.resize(observation_dim_ * num_elements);
}

/// \todo shared between analytical and numerical
template <class T>
T AnalyticalCost<T>::computeCost(const T* x) {
  model_->setup(x);

  for (int i = 0; i < num_elements_; ++i) {
    model_->f(&input_[i * input_dim_], &observations_[i * observation_dim_], &residual_data_[i * observation_dim_]);
  }

  return residual_data_.squaredNorm();
}

template <class T>
void AnalyticalCost<T>::computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) {
  model_->setup(x);

  int k = 0;
  for (int i = 0; i < num_elements_; ++i) {
    model_->f(&input_[i * input_dim_], &observations_[i * observation_dim_], &residual_data_[i * observation_dim_]);
    model_->df(&input_[i * input_dim_], &observations_[i * observation_dim_], jacobian_transposed_data_.data() + k);
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
