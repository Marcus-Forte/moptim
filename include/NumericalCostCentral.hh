#pragma once

#include <Eigen/Dense>
#include <mdspan>
#include <memory>
#include <span>

#include "ICost.hh"
#include "IModel.hh"

namespace moptim {

template <class T>
class NumericalCostCentral : public ICost<T> {
 public:
  NumericalCostCentral(const NumericalCostCentral&) = delete;

  ~NumericalCostCentral() override = default;

  NumericalCostCentral(std::mdspan<const T, std::dextents<size_t, 2>> input,
                       std::mdspan<const T, std::dextents<size_t, 2>> observations, size_t param_dim,
                       const std::shared_ptr<IModel<T>>& model);

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
  VectorT residual_data_minus_;

  std::mdspan<const T, std::dextents<size_t, 2>> input_elements_;
  std::mdspan<const T, std::dextents<size_t, 2>> observation_elements_;
  std::shared_ptr<IModel<T>> model_;
};

}  // namespace moptim