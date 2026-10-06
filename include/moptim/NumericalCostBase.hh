#pragma once

#include <Eigen/Dense>
#include <cassert>

#include "moptim/ICost.hh"
#include "moptim/IModel.hh"

namespace moptim {

/**
 * @brief Shared implementation for finite-difference numerical costs. `Derived` supplies the
 * finite-difference scheme by implementing a private `fillJacobian(x_vec, x_plus)` method that
 * fills `jacobian_data_` column by column (one column per parameter), using `callResiduals()` to
 * evaluate the model. `Derived` must befriend this base class.
 *
 * Holds the common state (input/observation buffers, model, residual/jacobian buffers) and
 * implements `computeCost()` and `computeLinearSystem()`, including the robustified (information
 * matrix / weights / robust loss) assembly, which is otherwise identical across finite-difference
 * schemes.
 */
template <class Derived, class Model, class T>
  requires NumericalModel<Model, T>
class NumericalCostBase : public ICost<T> {
 public:
  NumericalCostBase(const NumericalCostBase&) = delete;

  ~NumericalCostBase() override = default;

  NumericalCostBase(const T* input, const T* observations, size_t num_elements, size_t input_dim,
                    size_t observation_dim, size_t param_dim, Model model, size_t active_param_dim = 0)
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements, active_param_dim),
        input_elements_(input),
        observation_elements_(observations),
        model_(std::move(model)) {
    jacobian_data_.resize(observation_dim_ * num_elements_, active_param_dim_);
    residual_data_.resize(observation_dim_ * num_elements_);
    residual_data_plus_.resize(observation_dim_ * num_elements_);
  }

  T computeCost(const T* x) override {
    model_.setState(x);
    for (size_t i = 0; i < num_elements_; ++i) {
      model_.residual(x, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                      &residual_data_[i * observation_dim_]);
    }

    if (!isRobustified()) {
      return residual_data_.squaredNorm();
    }

    T cost = T{0};
    for (size_t i = 0; i < num_elements_; ++i) {
      auto residual_block = residual_data_.segment(i * observation_dim_, observation_dim_);
      cost += whitenAndScaleResidual(i, residual_block);
    }
    return cost;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    // Compute residuals at x
    callResiduals(x, residual_data_.data());

    Eigen::Map<const VectorT> x_vec(x, param_dim_);
    VectorT x_plus(x_vec);

    static_cast<Derived*>(this)->fillJacobian(x_vec, x_plus);

    Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb, param_dim_);
    // Only the leading active_param_dim_ block is populated; the trailing
    // rows/columns (parameters this cost does not depend on) stay zero.
    auto JTJ_active = JTJ_map.topLeftCorner(active_param_dim_, active_param_dim_);

    if (!isRobustified()) {
      JTJ_map.setZero();
      JTJ_active.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_data_.adjoint());
      JTJ_active = JTJ_active.template selfadjointView<Eigen::Lower>();
      JTb_map.setZero();
      JTb_map.head(active_param_dim_).noalias() = jacobian_data_.transpose() * residual_data_;
      cost = residual_data_.squaredNorm();
      return;
    }

    cost = T{0};
    for (size_t i = 0; i < num_elements_; ++i) {
      auto residual_block = residual_data_.segment(i * observation_dim_, observation_dim_);
      auto jacobian_block = jacobian_data_.block(i * observation_dim_, 0, observation_dim_, active_param_dim_);
      cost += whitenAndScaleElement(i, residual_block, jacobian_block, /*jacobian_rows_are_residuals=*/true);
    }

    // J^T*J is symmetric: compute only the lower triangle via rankUpdate (~2x fewer FLOPs),
    // then reflect to fill the full matrix.
    JTJ_map.setZero();
    JTJ_active.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_data_.adjoint());
    JTJ_active = JTJ_active.template selfadjointView<Eigen::Lower>();
    JTb_map.setZero();
    JTb_map.head(active_param_dim_).noalias() = jacobian_data_.transpose() * residual_data_;
  }

 protected:
  void callResiduals(const T* params, T* residual_out) {
    model_.setState(params);
    for (size_t i = 0; i < num_elements_; ++i) {
      model_.residual(params, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                      &residual_out[i * observation_dim_]);
    }
  }

  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;
  using ICost<T>::active_param_dim_;
  using ICost<T>::isRobustified;
  using ICost<T>::whitenAndScaleResidual;
  using ICost<T>::whitenAndScaleElement;

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT jacobian_data_;
  VectorT residual_data_;
  VectorT residual_data_plus_;

  const T* __restrict__ input_elements_;
  const T* __restrict__ observation_elements_;
  Model model_;
};

}  // namespace moptim
