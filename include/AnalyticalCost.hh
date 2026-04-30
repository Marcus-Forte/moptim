#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <mdspan>

#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class Model, class T>
  requires AnalyticalModel<Model, T>
class AnalyticalCost : public ICost<T> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;

  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 size_t param_dim, Model model = Model{})
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
        input_elements_(input, num_elements, input_dim),
        observation_elements_(observations, num_elements, observation_dim),
        model_(std::move(model)) {
    // We fill the jacobian transposed already
    jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements_);
    residual_data_.resize(observation_dim_ * num_elements_);
    jac_elem_buf_.resize(observation_dim_ * param_dim_);
  }

  T computeCost(const T* x) override {
    model_.setState(x);
    for (size_t i = 0; i < num_elements_; ++i) {
      model_.residual(x, &input_elements_[i, 0], &observation_elements_[i, 0], &residual_data_[i * observation_dim_]);
    }
    return residual_data_.squaredNorm();
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    model_.setState(x);
    for (size_t i = 0; i < num_elements_; ++i) {
      const T* in_i = &input_elements_[i, 0];
      const T* obs_i = &observation_elements_[i, 0];

      model_.residual(x, in_i, obs_i, &residual_data_[i * observation_dim_]);

      // Column-major: element i occupies observation_dim_ consecutive columns starting at i*observation_dim_
      model_.jacobian(x, in_i, obs_i, jacobian_transposed_data_.col(i * observation_dim_).data());
    }

    Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

    // jacobian_transposed_data_ stores J^T (param_dim x n_residuals).
    // rankUpdate(u) computes u*u^T, so rankUpdate(J^T) = J^T*J = JTJ.
    JTJ_map.setZero();
    JTJ_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_transposed_data_);
    JTJ_map = JTJ_map.template selfadjointView<Eigen::Lower>();
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
  VectorT jac_elem_buf_;

  std::mdspan<const T, std::dextents<size_t, 2>> input_elements_;
  std::mdspan<const T, std::dextents<size_t, 2>> observation_elements_;
  Model model_;
};

}  // namespace moptim
