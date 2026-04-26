#include "NumericalCostForwardEuler.hh"

#include <cmath>

namespace moptim {

template <class T>
NumericalCostForwardEuler<T>::NumericalCostForwardEuler(std::span<const T> input, std::span<const T> observations, size_t input_dim,
                                                        size_t observation_dim, size_t param_dim, size_t num_elements,
                                                        const std::shared_ptr<IModel<T>>& model)
    : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
      input_(input),
      observations_(observations),
      model_(model) {
  jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
  residual_data_.resize(observation_dim_ * num_elements_);
  residual_data_plus_.resize(observation_dim_ * num_elements);
}

/// \todo shared between analytical and numerical
/// \todo Eigen::Map?
template <class T>
T NumericalCostForwardEuler<T>::computeCost(std::span<const T> x) {
  model_->setup(x);

  for (int i = 0; i < num_elements_; ++i) {
    model_->f(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(i * observation_dim_, observation_dim_),
              std::span<T>(residual_data_.data() + i * observation_dim_, observation_dim_));
  }

  return residual_data_.squaredNorm();
}

template <class T>
void NumericalCostForwardEuler<T>::computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb,
                                                       T& cost) {
  model_->setup(x);

  // Compute residuals
  for (int i = 0; i < num_elements_; ++i) {
    model_->f(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(i * observation_dim_, observation_dim_),
              std::span<T>(residual_data_.data() + i * observation_dim_, observation_dim_));
  }

  Eigen::Map<const VectorT> x_vec(x.data(), param_dim_);

  const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());

  for (int i = 0; i < param_dim_; ++i) {
    VectorT x_plus(x_vec);
    x_plus[i] += g_step;

    model_->setup(std::span<const T>(x_plus.data(), x_plus.size()));

    for (int j = 0; j < num_elements_; ++j) {
      model_->f(input_.subspan(j * input_dim_, input_dim_), observations_.subspan(j * observation_dim_, observation_dim_),
                std::span<T>(residual_data_plus_.data() + j * observation_dim_, observation_dim_));
    }

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_) / g_step;
  }

  Eigen::Map<MatrixT> JTJ_map(JTJ.data(), param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb.data(), param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_data_.transpose() * jacobian_data_;
  JTb_map = jacobian_data_.transpose() * residual_data_;
  cost = residual_data_.squaredNorm();
}

template class NumericalCostForwardEuler<double>;
template class NumericalCostForwardEuler<float>;

}  // namespace moptim