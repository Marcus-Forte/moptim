#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#include "moptim/IModel.hh"
#include "moptim/NumericalCostBase.hh"

namespace moptim {

template <class Model, class T>
  requires NumericalModel<Model, T>
class NumericalCostForwardEuler : public NumericalCostBase<NumericalCostForwardEuler<Model, T>, Model, T> {
  using Base = NumericalCostBase<NumericalCostForwardEuler<Model, T>, Model, T>;
  friend Base;

 public:
  NumericalCostForwardEuler(const NumericalCostForwardEuler&) = delete;

  ~NumericalCostForwardEuler() override = default;

  NumericalCostForwardEuler(const T* input, const T* observations, size_t num_elements, size_t input_dim,
                            size_t observation_dim, size_t param_dim, Model model = Model{},
                            size_t active_param_dim = 0)
      : Base(input, observations, num_elements, input_dim, observation_dim, param_dim, std::move(model),
             active_param_dim) {}

 private:
  using VectorT = typename Base::VectorT;

  // Fills jacobian_data_ one column per active parameter using a forward (one-sided) finite difference.
  void fillJacobian(const VectorT& x_vec, VectorT& x_plus) {
    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
    const T inv_g_step = T{1} / g_step;

    for (size_t i = 0; i < this->active_param_dim_; ++i) {
      x_plus[i] = x_vec[i] + g_step;

      this->callResiduals(x_plus.data(), this->residual_data_plus_.data());

      x_plus[i] = x_vec[i];

      this->jacobian_data_.col(i) = (this->residual_data_plus_ - this->residual_data_) * inv_g_step;
    }
  }
};

}  // namespace moptim
