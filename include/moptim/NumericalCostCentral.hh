#pragma once

#include <Eigen/Dense>
#include <cassert>

#include "moptim/IModel.hh"
#include "moptim/NumericalCostBase.hh"

namespace moptim {

template <class Model, class T>
  requires NumericalModel<Model, T>
class NumericalCostCentral : public NumericalCostBase<NumericalCostCentral<Model, T>, Model, T> {
  using Base = NumericalCostBase<NumericalCostCentral<Model, T>, Model, T>;
  friend Base;

 public:
  NumericalCostCentral(const NumericalCostCentral&) = delete;

  ~NumericalCostCentral() override = default;

  NumericalCostCentral(const T* input, const T* observations, size_t num_elements, size_t input_dim,
                       size_t observation_dim, size_t param_dim, Model model = Model{})
      : Base(input, observations, num_elements, input_dim, observation_dim, param_dim, std::move(model)) {
    residual_data_minus_.resize(observation_dim * num_elements);
  }

 private:
  using VectorT = typename Base::VectorT;

  // Fills jacobian_data_ one column per parameter using a central (symmetric) finite difference.
  void fillJacobian(const VectorT& x_vec, VectorT& x_plus) {
    const T g_step = std::sqrt(std::numeric_limits<T>::epsilon());
    const T inv_2g_step = T{1} / (T{2} * g_step);

    for (size_t i = 0; i < this->param_dim_; ++i) {
      x_plus[i] = x_vec[i] + g_step;
      this->callResiduals(x_plus.data(), this->residual_data_plus_.data());

      x_plus[i] = x_vec[i] - g_step;
      this->callResiduals(x_plus.data(), residual_data_minus_.data());

      x_plus[i] = x_vec[i];

      this->jacobian_data_.col(i) = (this->residual_data_plus_ - residual_data_minus_) * inv_2g_step;
    }
  }

  VectorT residual_data_minus_;  // reused as x_minus residuals
};

}  // namespace moptim
