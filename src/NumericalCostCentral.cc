#include "NumericalCostCentral.hh"

#include <cmath>

namespace moptim {

template <class T>
NumericalCostCentral<T>::NumericalCostCentral(const T* input, const T* observations, size_t input_dim,
                                              size_t observation_dim, size_t param_dim, size_t num_elements,
                                              const std::shared_ptr<IModel<T>>& model)
    : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
      input_(input),
      observations_(observations),
      model_(model) {
  jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
  residual_data_.resize(observation_dim_ * num_elements_);
  residual_data_plus_.resize(observation_dim_ * num_elements_);
  residual_data_minus_.resize(observation_dim_ * num_elements_);
}

/// \todo shared between analytical and numerical
/// \todo perhaps pass X dimensions at construction
/// \todo Eigen::Map?
template <class T>
T NumericalCostCentral<T>::computeCost(const T* x) {
  model_->setup(x);

  for (int i = 0; i < num_elements_; ++i) {
    model_->f(&input_[i * input_dim_], &observations_[i * observation_dim_], &residual_data_[i * observation_dim_]);
  }

  return residual_data_.squaredNorm();
}

template <class T>
void NumericalCostCentral<T>::computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) {
  model_->setup(x);

  // Compute residuals
  for (int i = 0; i < num_elements_; ++i) {
    model_->f(&input_[i * input_dim_], &observations_[i * observation_dim_], &residual_data_[i * observation_dim_]);
  }

  Eigen::Map<const VectorT> x_vec(x, param_dim_);

  const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());

  for (int i = 0; i < param_dim_; ++i) {
    VectorT x_plus(x_vec);
    VectorT x_minus(x_vec);
    x_plus[i] += g_step;
    x_minus[i] -= g_step;

    model_->setup(x_plus.data());
    for (int j = 0; j < num_elements_; ++j) {
      model_->f(&input_[j * input_dim_], &observations_[j * observation_dim_],
                &residual_data_plus_[j * observation_dim_]);
    }

    model_->setup(x_minus.data());
    for (int j = 0; j < num_elements_; ++j) {
      model_->f(&input_[j * input_dim_], &observations_[j * observation_dim_],
                &residual_data_minus_[j * observation_dim_]);
    }

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_minus_) / (2 * g_step);
  }
  Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_data_.transpose() * jacobian_data_;
  JTb_map = jacobian_data_.transpose() * residual_data_;
  *cost = residual_data_.squaredNorm();
}

template class NumericalCostCentral<double>;
template class NumericalCostCentral<float>;

}  // namespace moptim