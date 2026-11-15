#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <utility>

#include "moptim/ChunkedCost.hh"
#include "moptim/IModel.hh"

namespace moptim {

/**
 * @brief Cost whose jacobian comes from the model's own `jacobian()`.
 *
 * The model writes the derivative of its residuals with respect to the step `delta` at `delta = 0`
 * (see IModel.hh). With Euclidean parameters, built from `param_dim`, that is the ordinary
 * jacobian with respect to `x`. With a manifold it is the jacobian with respect to the tangent
 * vector of `x [+] delta`, so a pose is differentiated in the convention of the manifold's `plus`.
 */
template <class Model, class T>
  requires AnalyticalModel<Model, T>
class AnalyticalCost : public ChunkedCost<Model, T> {
  using Base = ChunkedCost<Model, T>;

 public:
  using typename Base::MatrixT;
  using typename Base::VectorT;

  /// Euclidean parameters: `x` has `param_dim` entries.
  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 size_t param_dim, Model model = Model{}, size_t active_param_dim = 0)
      : AnalyticalCost(input, observations, num_elements, input_dim, observation_dim, Base::euclideanSpace(param_dim),
                       std::move(model), active_param_dim) {}

  /// Parameters on a manifold. `active_param_dim` counts leading tangent dimensions.
  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 std::shared_ptr<const IManifold<T>> manifold, Model model = Model{}, size_t active_param_dim = 0)
      : AnalyticalCost(input, observations, num_elements, input_dim, observation_dim,
                       Base::spaceOf(std::move(manifold)), std::move(model), active_param_dim) {}

 private:
  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 typename Base::ParamSpace space, Model model, size_t active_param_dim)
      : Base(input, observations, num_elements, input_dim, observation_dim, std::move(space), std::move(model),
             active_param_dim),
        element_jacobian_(active_param_dim_ * observation_dim_) {}

  using Base::active_param_dim_;
  using Base::observation_dim_;

  void evaluateJacobian(const T* x, size_t first, size_t count, const Eigen::Ref<const VectorT>& /*residuals*/,
                        Eigen::Ref<MatrixT> jacobian) override {
    fillJacobian(x, first, count, element_jacobian_.data(), jacobian.data(), jacobian.outerStride());
  }

  /// `transposed` and `jacobian` are buffers of this cost that nothing else points into; see
  /// ChunkedCost::evaluateResiduals() for why they are declared so and why this is not inlined.
  [[gnu::noinline]] void fillJacobian(const T* x, size_t first, size_t count, T* __restrict__ transposed,
                                      T* __restrict__ jacobian, size_t jacobian_stride) {
    const size_t params = active_param_dim_;
    const size_t dim = observation_dim_;

    for (size_t e = 0; e < count; ++e) {
      this->model_.jacobian(x, this->input(first + e), this->observation(first + e), transposed);

      // The model writes the element's transposed jacobian; the chunk stores one column per parameter.
      for (size_t p = 0; p < params; ++p) {
        T* column = jacobian + p * jacobian_stride + e * dim;
        for (size_t i = 0; i < dim; ++i) {
          column[i] = transposed[p + params * i];
        }
      }
    }
  }

  /// Transposed jacobian of one element as the model writes it: active_param_dim x observation_dim,
  /// column-major.
  VectorT element_jacobian_;
};

}  // namespace moptim
