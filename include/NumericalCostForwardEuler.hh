#pragma once

#include <Eigen/Dense>
#include <memory>
#include <span>

#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class T>
class NumericalCostForwardEuler : public ICost<T> {
 public:
  NumericalCostForwardEuler(const NumericalCostForwardEuler&) = delete;

  ~NumericalCostForwardEuler() override = default;

  NumericalCostForwardEuler(std::span<const T> input, std::span<const T> observations, size_t input_dim, size_t observation_dim,
                            size_t param_dim, size_t num_elements, const std::shared_ptr<IModel<T>>& model);

  T computeCost(std::span<const T> x) override;

  void computeLinearSystem(std::span<const T> x, std::span<T> JTJ, std::span<T> JTb, T& cost) override;

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

  std::span<const T> input_;
  std::span<const T> observations_;
  std::shared_ptr<IModel<T>> model_;
};

}  // namespace moptim