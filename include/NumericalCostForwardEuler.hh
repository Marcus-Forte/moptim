#pragma once

#include <Eigen/Dense>
#include <memory>

#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class T>
class NumericalCostForwardEuler : public ICost<T> {
 public:
  NumericalCostForwardEuler(const NumericalCostForwardEuler&) = delete;

  ~NumericalCostForwardEuler() override = default;

  NumericalCostForwardEuler(const T* input, const T* observations, size_t num_elements, size_t output_dim,
                            size_t param_dim, const std::shared_ptr<IModel<T>>& model);

  T computeCost(const T* x) override;

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) override;

 private:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT jacobian_data_;
  VectorT residual_data_;
  VectorT residual_data_plus_;

  using ICost<T>::num_elements_;
  using ICost<T>::dimensions_;

  const T* input_;
  const T* observations_;
  const size_t output_dim_;
  const size_t param_dim_;
  const size_t residuals_dim_;
  std::shared_ptr<IModel<T>> model_;
};

}  // namespace moptim