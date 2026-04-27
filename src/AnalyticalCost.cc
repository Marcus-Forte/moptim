#include "AnalyticalCost.hh"

#include <cassert>
#include <ranges>

#include "CostComputeUtils.hh"

namespace moptim {
template <class T>
AnalyticalCost<T>::AnalyticalCost(std::mdspan<const T, std::dextents<size_t, 2>> input,
                                  std::mdspan<const T, std::dextents<size_t, 2>> observations, size_t param_dim,
                                  const std::shared_ptr<IJacobianModel<T>>& model)
    : ICost<T>(input.extent(1), observations.extent(1), param_dim, input.extent(0)),
      input_elements_(input),
      observation_elements_(observations),
      model_{model} {
  // We fill the jacobian transposed already
  jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements_);
  residual_data_.resize(observation_dim_ * num_elements_);
}

template <class T>
T AnalyticalCost<T>::computeCost(std::span<const T> x) {
  return detail::computeCost(x, param_dim_, *model_, input_elements_, observation_elements_,
                             std::span<T>(residual_data_.data(), residual_data_.size()));
}

template <class T>
void AnalyticalCost<T>::computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) {
  assert(x.size() == param_dim_);
  assert(JTJ.size() == param_dim_ * param_dim_);
  assert(JTb.size() == param_dim_);

  model_->setup(x);

  const auto* input_data = input_elements_.data_handle();
  const auto* observation_data = observation_elements_.data_handle();
  size_t k = 0;
  for (const size_t i : std::views::iota(size_t{0}, input_elements_.extent(0))) {
    const auto row = i * observation_dim_;
    model_->f(std::span<const T>(input_data + (i * input_dim_), input_dim_),
              std::span<const T>(observation_data + row, observation_dim_),
              std::span<T>(residual_data_.data() + row, observation_dim_));
    model_->df(std::span<const T>(input_data + (i * input_dim_), input_dim_),
               std::span<const T>(observation_data + row, observation_dim_),
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
