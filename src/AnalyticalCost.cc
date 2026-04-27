#include "AnalyticalCost.hh"
#include "CostSizeUtils.hh"

#include <cassert>

namespace moptim {
template <class T>
AnalyticalCost<T>::AnalyticalCost(std::span<const T> input, std::span<const T> observations, size_t input_dim,
                                  size_t observation_dim, size_t param_dim,
                                  const std::shared_ptr<IJacobianModel<T>>& model)
    : ICost<T>(input_dim, observation_dim, param_dim,
               detail::inferNumElements(input, observations, input_dim, observation_dim)),
      input_{input},
      observations_{observations},
      model_{model} {
  // We fill the jacobian transposed already
  jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements_);
  residual_data_.resize(observation_dim_ * num_elements_);
}

/// \todo shared between analytical and numerical
template <class T>
T AnalyticalCost<T>::computeCost(std::span<const T> x) {
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
void AnalyticalCost<T>::computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) {
  assert(x.size() == param_dim_);
  assert(JTJ.size() == param_dim_ * param_dim_);
  assert(JTb.size() == param_dim_);

  model_->setup(x);

  size_t k = 0;
  for (size_t i = 0; i < num_elements_; ++i) {
    const auto row = i * observation_dim_;
    model_->f(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(row, observation_dim_),
              std::span<T>(residual_data_.data() + row, observation_dim_));
    model_->df(input_.subspan(i * input_dim_, input_dim_), observations_.subspan(row, observation_dim_),
               std::span<T>(jacobian_transposed_data_.data() + k, param_dim_ * observation_dim_));
    k += param_dim_ * observation_dim_;
  }

  Eigen::Map<MatrixT> JTJ_map(JTJ.data(), param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb.data(), param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_transposed_data_ * jacobian_transposed_data_.transpose();
  JTb_map = jacobian_transposed_data_ * residual_data_;
  cost = residual_data_.squaredNorm();
}

template class AnalyticalCost<double>;
template class AnalyticalCost<float>;

}  // namespace moptim
