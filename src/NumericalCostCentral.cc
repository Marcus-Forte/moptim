#include "NumericalCostCentral.hh"

#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>

#include "CostComputeUtils.hh"

namespace moptim {

template <class T>
NumericalCostCentral<T>::NumericalCostCentral(std::mdspan<const T, std::dextents<size_t, 2>> input,
                                              std::mdspan<const T, std::dextents<size_t, 2>> observations,
                                              size_t param_dim, const std::shared_ptr<IModel<T>>& model)
    : ICost<T>(input.extent(1), observations.extent(1), param_dim, input.extent(0)),
      input_elements_(input),
      observation_elements_(observations),
      model_(model) {
  jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
  residual_data_.resize(observation_dim_ * num_elements_);
  residual_data_plus_.resize(observation_dim_ * num_elements_);
  residual_data_minus_.resize(observation_dim_ * num_elements_);
}

/// \todo perhaps pass X dimensions at construction
/// \todo Eigen::Map?
template <class T>
T NumericalCostCentral<T>::computeCost(std::span<const T> x) {
  return detail::computeCost(x, param_dim_, *model_, input_elements_, observation_elements_,
                             std::span<T>(residual_data_.data(), residual_data_.size()));
}

template <class T>
void NumericalCostCentral<T>::computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) {
  assert(x.size() == param_dim_);
  assert(JTJ.size() == param_dim_ * param_dim_);
  assert(JTb.size() == param_dim_);

  model_->setup(x);

  const auto computeResiduals = [this](std::span<T> residual_out) {
    const auto* input_data = input_elements_.data_handle();
    const auto* observation_data = observation_elements_.data_handle();
    for (const size_t i : std::views::iota(size_t{0}, input_elements_.extent(0))) {
      const auto row = i * observation_dim_;
      model_->f(std::span<const T>(input_data + (i * input_dim_), input_dim_),
                std::span<const T>(observation_data + row, observation_dim_),
                residual_out.subspan(row, observation_dim_));
    }
  };

  // Compute residuals
  computeResiduals(std::span<T>(residual_data_.data(), residual_data_.size()));

  Eigen::Map<const VectorT> x_vec(x.data(), param_dim_);
  VectorT x_plus(x_vec);
  VectorT x_minus(x_vec);

  const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
  const T inv_2g_step = T{1} / (T{2} * g_step);

  for (size_t i = 0; i < param_dim_; ++i) {
    x_plus[i] = x_vec[i] + g_step;
    x_minus[i] = x_vec[i] - g_step;

    model_->setup(std::span<const T>(x_plus.data(), x_plus.size()));
    computeResiduals(std::span<T>(residual_data_plus_.data(), residual_data_plus_.size()));

    model_->setup(std::span<const T>(x_minus.data(), x_minus.size()));
    computeResiduals(std::span<T>(residual_data_minus_.data(), residual_data_minus_.size()));

    x_plus[i] = x_vec[i];
    x_minus[i] = x_vec[i];

    jacobian_data_.col(i) = (residual_data_plus_ - residual_data_minus_) * inv_2g_step;
  }
  Eigen::Map<MatrixT> JTJ_map(JTJ.data(), param_dim_, param_dim_);
  Eigen::Map<VectorT> JTb_map(JTb.data(), param_dim_);

  /// \todo can use rank update?
  JTJ_map = jacobian_data_.transpose() * jacobian_data_;
  JTb_map = jacobian_data_.transpose() * residual_data_;
  cost = residual_data_.squaredNorm();
}

template class NumericalCostCentral<double>;
template class NumericalCostCentral<float>;

}  // namespace moptim