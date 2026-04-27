#include "NumericalCostForwardEuler.hh"
#include "CostSizeUtils.hh"

#include <cassert>
#include <cmath>
#include <limits>

namespace moptim {

template <class T>
NumericalCostForwardEuler<T>::NumericalCostForwardEuler(std::span<const T> input, std::span<const T> observations, size_t input_dim,
                                                        size_t observation_dim, size_t param_dim,
                                                        const std::shared_ptr<IModel<T>>& model)
    : ICost<T>(input_dim, observation_dim, param_dim,
               detail::inferNumElements(input, observations, input_dim, observation_dim)),
      input_(input),
      observations_(observations),
      model_(model) {
  jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
  residual_data_.resize(observation_dim_ * num_elements_);
  residual_data_plus_.resize(observation_dim_ * num_elements_);
}

/// \todo shared between analytical and numerical
/// \todo Eigen::Map?
template <class T>
T NumericalCostForwardEuler<T>::computeCost(std::span<const T> x) {
  assert(x.size() == param_dim_);

  model_->setup(x);

  for (size_t i = 0; i < num_elements_; ++i) {
    const auto row = i * observation_dim_;
    model_->f(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(row, observation_dim_),
              std::span<T>(residual_data_.data() + row, observation_dim_));
  }

  return residual_data_.squaredNorm();
}

template <class T>
void NumericalCostForwardEuler<T>::computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb,
                                                       T& cost) {
  assert(x.size() == param_dim_);
  assert(JTJ.size() == param_dim_ * param_dim_);
  assert(JTb.size() == param_dim_);

  model_->setup(x);

  const auto computeResiduals = [this](std::span<T> residual_out) {
    for (size_t i = 0; i < num_elements_; ++i) {
      const auto row = i * observation_dim_;
      model_->f(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(row, observation_dim_),
                residual_out.subspan(row, observation_dim_));
    }
  };

  // Compute residuals
  computeResiduals(std::span<T>(residual_data_.data(), residual_data_.size()));

  Eigen::Map<const VectorT> x_vec(x.data(), param_dim_);
  VectorT x_plus(x_vec);

  const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
  const T inv_g_step = T{1} / g_step;

  for (size_t i = 0; i < param_dim_; ++i) {
    x_plus[i] = x_vec[i] + g_step;

    model_->setup(std::span<const T>(x_plus.data(), x_plus.size()));
    computeResiduals(std::span<T>(residual_data_plus_.data(), residual_data_plus_.size()));

    x_plus[i] = x_vec[i];

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_) * inv_g_step;
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