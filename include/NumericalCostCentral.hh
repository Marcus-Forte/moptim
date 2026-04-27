#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <limits>
#include <mdspan>
#include <span>

#include "CostComputeUtils.hh"
#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class Model, class T>
class NumericalCostCentral : public ICost<T> {
 public:
  NumericalCostCentral(const NumericalCostCentral&) = delete;

  ~NumericalCostCentral() override = default;

  NumericalCostCentral(std::mdspan<const T, std::dextents<size_t, 2>> input,
                       std::mdspan<const T, std::dextents<size_t, 2>> observations, size_t param_dim,
                       Model model)
      : ICost<T>(input.extent(1), observations.extent(1), param_dim, input.extent(0)),
        input_elements_(input),
        observation_elements_(observations),
        model_(std::move(model)) {
    jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
    residual_data_.resize(observation_dim_ * num_elements_);
    residual_data_plus_.resize(observation_dim_ * num_elements_);
    residual_data_minus_.resize(observation_dim_ * num_elements_);
  }

  T computeCost(std::span<const T> x) override {
    return detail::computeCost(x, param_dim_, model_, input_elements_, observation_elements_,
                               std::span<T>(residual_data_.data(), residual_data_.size()));
  }

  void computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) override {
    assert(x.size() == param_dim_);
    assert(JTJ.size() == param_dim_ * param_dim_);
    assert(JTb.size() == param_dim_);

    const auto computeResiduals = [this](std::span<const T> params, std::span<T> residual_out) {
      std::mdspan residuals_md(residual_out.data(), num_elements_, observation_dim_);
      model_.residuals(params, input_elements_, observation_elements_, residuals_md);
    };

    // Compute residuals at x
    computeResiduals(x, std::span<T>(residual_data_.data(), residual_data_.size()));

    Eigen::Map<const VectorT> x_vec(x.data(), param_dim_);
    VectorT x_plus(x_vec);
    VectorT x_minus(x_vec);

    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
    const T inv_2g_step = T{1} / (T{2} * g_step);

    for (size_t i = 0; i < param_dim_; ++i) {
      x_plus[i] = x_vec[i] + g_step;
      x_minus[i] = x_vec[i] - g_step;

      model_.residuals(std::span<const T>(x_plus.data(), x_plus.size()), input_elements_, observation_elements_,
                       std::mdspan(residual_data_plus_.data(), num_elements_, observation_dim_));

      model_.residuals(std::span<const T>(x_minus.data(), x_minus.size()), input_elements_, observation_elements_,
                       std::mdspan(residual_data_minus_.data(), num_elements_, observation_dim_));

      x_plus[i] = x_vec[i];
      x_minus[i] = x_vec[i];

      jacobian_data_.col(i) = (residual_data_plus_ - residual_data_minus_) * inv_2g_step;
    }

    Eigen::Map<MatrixT> JTJ_map(JTJ.data(), param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb.data(), param_dim_);

    JTJ_map.noalias() = jacobian_data_.transpose() * jacobian_data_;
    JTb_map.noalias() = jacobian_data_.transpose() * residual_data_;
    cost = residual_data_.squaredNorm();
  }

 private:
  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT jacobian_data_;
  VectorT residual_data_;
  VectorT residual_data_plus_;
  VectorT residual_data_minus_;

  std::mdspan<const T, std::dextents<size_t, 2>> input_elements_;
  std::mdspan<const T, std::dextents<size_t, 2>> observation_elements_;
  Model model_;
};

}  // namespace moptim
