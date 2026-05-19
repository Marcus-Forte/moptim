#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#include "moptim/ICost.h"
#include "moptim/IModel.h"
#include "moptim/PlusOperations/EuclideanPlusOperator.h"
#include "moptim/PlusOperations/IPlus.h"

namespace moptim {

template <class Model, class T,
          class PlusOperator = EuclideanPlusOperator<T>>
  requires(IPlus<PlusOperator, T> && NumericalModel<Model, T>)
class NumericalCostForwardEuler : public ICost<T> {
 public:
  NumericalCostForwardEuler(const NumericalCostForwardEuler&) = delete;

  ~NumericalCostForwardEuler() override = default;

  NumericalCostForwardEuler(const T* input, const T* observations, size_t num_elements, size_t input_dim,
                            size_t observation_dim, size_t param_dim, Model model = Model{})
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
        input_elements_(input),
        observation_elements_(observations),
        model_(std::move(model)) {
    jacobian_data_.resize(observation_dim_ * num_elements_, param_dim_);
    residual_data_.resize(observation_dim_ * num_elements_);
    residual_data_plus_.resize(observation_dim_ * num_elements_);
    x_plus_.resize(param_dim_);
    delta_.resize(param_dim_);
  }

  T computeCost(const T* x) override {
    model_.setState(x);
    for (size_t i = 0; i < num_elements_; ++i) {
      model_.residual(x, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                      &residual_data_[i * observation_dim_]);
    }
    return residual_data_.squaredNorm();
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    const auto callResiduals = [this](const T* params, T* residual_out) {
      model_.setState(params);
      for (size_t i = 0; i < num_elements_; ++i) {
        model_.residual(params, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                        &residual_out[i * observation_dim_]);
      }
    };

    // Compute residuals
    callResiduals(x, residual_data_.data());

    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
    const T inv_g_step = T{1} / g_step;

    for (size_t i = 0; i < param_dim_; ++i) {
      delta_.setZero();
      delta_[i] = g_step;
      PlusOperator::plus(x, delta_.data(), x_plus_.data(), param_dim_);

      callResiduals(x_plus_.data(), residual_data_plus_.data());

      jacobian_data_.col(i) = (residual_data_plus_ - residual_data_) * inv_g_step;
    }

    Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

    // J^T*J is symmetric: compute only the lower triangle via rankUpdate (~2x fewer FLOPs),
    // then reflect to fill the full matrix.
    JTJ_map.setZero();
    JTJ_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_data_.adjoint());
    JTJ_map = JTJ_map.template selfadjointView<Eigen::Lower>();
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
  VectorT x_plus_;
  VectorT delta_;

  const T* input_elements_;
  const T* observation_elements_;
  Model model_;
};

}  // namespace moptim
