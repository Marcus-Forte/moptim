#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <mdspan>
#include <span>

#include "CostComputeUtils.hh"
#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class Model, class T>
class AnalyticalCost : public ICost<T> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;

  AnalyticalCost(std::mdspan<const T, std::dextents<size_t, 2>> input,
                 std::mdspan<const T, std::dextents<size_t, 2>> observations, size_t param_dim,
                 Model model)
      : ICost<T>(input.extent(1), observations.extent(1), param_dim, input.extent(0)),
        input_elements_(input),
        observation_elements_(observations),
        model_(std::move(model)) {
    // We fill the jacobian transposed already
    jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements_);
    residual_data_.resize(observation_dim_ * num_elements_);
  }

  T computeCost(std::span<const T> x) override {
    return detail::computeCost(x, param_dim_, model_, input_elements_, observation_elements_,
                               std::span<T>(residual_data_.data(), residual_data_.size()));
  }

  void computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) override {
    assert(x.size() == param_dim_);
    assert(JTJ.size() == param_dim_ * param_dim_);
    assert(JTb.size() == param_dim_);

    // Residuals
    std::mdspan residuals_md(residual_data_.data(), num_elements_, observation_dim_);
    model_.residuals(x, input_elements_, observation_elements_, residuals_md);

    // Jacobians: layout [num_elements x (observation_dim * param_dim)]
    std::mdspan jacobians_md(jacobian_transposed_data_.data(), num_elements_, observation_dim_ * param_dim_);
    model_.jacobians(x, input_elements_, observation_elements_, jacobians_md);

    Eigen::Map<MatrixT> JTJ_map(JTJ.data(), param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb.data(), param_dim_);

    JTJ_map.noalias() = jacobian_transposed_data_ * jacobian_transposed_data_.transpose();
    JTb_map.noalias() = jacobian_transposed_data_ * residual_data_;
    cost = residual_data_.squaredNorm();
  }

 private:
  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT jacobian_transposed_data_;
  VectorT residual_data_;

  std::mdspan<const T, std::dextents<size_t, 2>> input_elements_;
  std::mdspan<const T, std::dextents<size_t, 2>> observation_elements_;
  Model model_;
};

}  // namespace moptim
