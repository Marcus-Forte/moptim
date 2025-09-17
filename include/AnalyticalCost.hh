#pragma once

#include <Eigen/Dense>
#include <memory>

#include "ICost.hh"
#include "IModel.hh"

namespace moptim {
template <class T>
class AnalyticalCost : public ICost<T> {
 public:
  AnalyticalCost(const AnalyticalCost&) = delete;

  AnalyticalCost<T>(const T* input, const T* observations, size_t input_dim, size_t observation_dim, size_t param_dim,
                    size_t num_elements, const std::shared_ptr<IJacobianModel<T>>& model);

  T computeCost(const T* x) override;

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) override;

 private:
  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::num_elements_;

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  MatrixT jacobian_transposed_data_;
  VectorT residual_data_;

  const T* input_;
  const T* observations_;
  
  std::shared_ptr<IJacobianModel<T>> model_;
};
}  // namespace moptim