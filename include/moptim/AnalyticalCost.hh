#pragma once

#include <Eigen/Dense>
#include <cassert>

#include "moptim/ICost.hh"
#include "moptim/IModel.hh"

namespace moptim {

template <class Model, class T>
  requires AnalyticalModel<Model, T>
class AnalyticalCost : public ICost<T> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;

  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 size_t param_dim, Model model = Model{})
      : ICost<T>(input_dim, observation_dim, param_dim, num_elements),
        input_elements_(input),
        observation_elements_(observations),
        model_(std::move(model)) {
    // We fill the jacobian transposed already
    jacobian_transposed_data_.resize(param_dim_, observation_dim_ * num_elements_);
    residual_data_.resize(observation_dim_ * num_elements_);
    jac_elem_buf_.resize(observation_dim_ * param_dim_);
  }

  T computeCost(const T* x) override {
    model_.setState(x);

    if (!isRobustified()) {
      for (size_t i = 0; i < num_elements_; ++i) {
        model_.residual(x, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                        &residual_data_[i * observation_dim_]);
      }
      return residual_data_.squaredNorm();
    }

    T cost = T{0};
    for (size_t i = 0; i < num_elements_; ++i) {
      auto residual_block = residual_data_.segment(i * observation_dim_, observation_dim_);
      model_.residual(x, input_elements_ + i * input_dim_, observation_elements_ + i * observation_dim_,
                      residual_block.data());

      if (hasInformation()) {
        residual_block = informationFactor(i).transpose() * residual_block;
      } else if (hasWeights()) {
        residual_block *= std::sqrt(weight(i));
      }

      T contribution;
      applyLoss(residual_block.squaredNorm(), contribution);
      cost += contribution;
    }
    return cost;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    model_.setState(x);

    Eigen::Map<MatrixT> JTJ_map(JTJ, param_dim_, param_dim_);
    Eigen::Map<VectorT> JTb_map(JTb, param_dim_);

    if (!isRobustified()) {
      for (size_t i = 0; i < num_elements_; ++i) {
        const T* in_i = input_elements_ + i * input_dim_;
        const T* obs_i = observation_elements_ + i * observation_dim_;

        model_.residual(x, in_i, obs_i, &residual_data_[i * observation_dim_]);
        // Column-major: element i occupies observation_dim_ consecutive columns starting at i*observation_dim_
        model_.jacobian(x, in_i, obs_i, jacobian_transposed_data_.col(i * observation_dim_).data());
      }

      JTJ_map.setZero();
      JTJ_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_transposed_data_);
      JTJ_map = JTJ_map.template selfadjointView<Eigen::Lower>();
      JTb_map.noalias() = jacobian_transposed_data_ * residual_data_;
      cost = residual_data_.squaredNorm();
      return;
    }

    cost = T{0};
    for (size_t i = 0; i < num_elements_; ++i) {
      const T* in_i = input_elements_ + i * input_dim_;
      const T* obs_i = observation_elements_ + i * observation_dim_;

      auto residual_block = residual_data_.segment(i * observation_dim_, observation_dim_);
      model_.residual(x, in_i, obs_i, residual_block.data());

      // Column-major: element i occupies observation_dim_ consecutive columns starting at i*observation_dim_
      auto jacobian_block = jacobian_transposed_data_.block(0, i * observation_dim_, param_dim_, observation_dim_);
      model_.jacobian(x, in_i, obs_i, jacobian_block.data());

      // jacobian_block stores J_i^T (param_dim x observation_dim), so whitening/scaling J_i on the left
      // translates to scaling J_i^T on the right.
      if (hasInformation()) {
        const MatrixT& L = informationFactor(i);
        residual_block = L.transpose() * residual_block;
        jacobian_block = (jacobian_block * L).eval();
      } else if (hasWeights()) {
        const T w = std::sqrt(weight(i));
        residual_block *= w;
        jacobian_block *= w;
      }

      T contribution;
      const T scale = applyLoss(residual_block.squaredNorm(), contribution);
      residual_block *= scale;
      jacobian_block *= scale;
      cost += contribution;
    }

    // jacobian_transposed_data_ stores J^T (param_dim x n_residuals).
    // rankUpdate(u) computes u*u^T, so rankUpdate(J^T) = J^T*J = JTJ.
    JTJ_map.setZero();
    JTJ_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_transposed_data_);
    JTJ_map = JTJ_map.template selfadjointView<Eigen::Lower>();
    JTb_map.noalias() = jacobian_transposed_data_ * residual_data_;
  }

 private:
  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;
  using ICost<T>::hasInformation;
  using ICost<T>::hasWeights;
  using ICost<T>::isRobustified;
  using ICost<T>::informationFactor;
  using ICost<T>::weight;
  using ICost<T>::applyLoss;

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT jacobian_transposed_data_;
  VectorT residual_data_;
  VectorT jac_elem_buf_;

  const T* input_elements_;
  const T* observation_elements_;
  Model model_;
};

}  // namespace moptim
