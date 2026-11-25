#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "moptim/ChunkedCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/IModel.hh"
#include "moptim/detail/ChunkKernel.hh"

namespace moptim {
namespace detail {

/**
 * Jacobian policy of AnalyticalCost: the rows of a chunk come from the model's own `jacobian()`.
 * Resolved at compile time, so the model's call is inlined into the loop that scatters its result.
 */
template <class Model, class T>
struct AnalyticalJacobian {
  /// Both member functions can be called on a const model, so the workers can share the one model.
  static constexpr bool kConstCallable = ConstResidualModel<Model, T> && ConstJacobianModel<Model, T>;

  /// Evaluation points besides `x` per active parameter: none, the model gives the jacobian.
  static constexpr size_t kPointsPerParam = 0;

  /// Values of scratch the policy needs: the transposed jacobian of one element.
  static constexpr size_t auxValues(size_t /*chunk_rows*/, size_t params, size_t observation_dim) {
    return params * observation_dim;
  }

  /// Writes the jacobian rows of elements [first, first + count): column `p` of the chunk (column
  /// stride `jacobian_stride`) holds the derivatives with respect to step entry `p`.
  ///
  /// Dispatches on the observation dimension to evaluateJacobianOfDim(), which does the work. See
  /// there for why the dimension is a template parameter, and evaluateResiduals() for why `transposed`
  /// and `jacobian` are declared `__restrict__` and the loop is kept out of line.
  template <class ModelPtr>
  static void evaluateJacobian(ModelPtr models, const KernelArgs<T>& args, size_t first, size_t count,
                               const T* residuals, T* __restrict__ transposed, T* __restrict__ jacobian,
                               size_t jacobian_stride) {
    switch (args.observation_dim) {
      case 1:
        return evaluateJacobianOfDim<1>(models, args, first, count, residuals, transposed, jacobian, jacobian_stride);
      case 2:
        return evaluateJacobianOfDim<2>(models, args, first, count, residuals, transposed, jacobian, jacobian_stride);
      case 3:
        return evaluateJacobianOfDim<3>(models, args, first, count, residuals, transposed, jacobian, jacobian_stride);
      default:
        return evaluateJacobianOfDim<0>(models, args, first, count, residuals, transposed, jacobian, jacobian_stride);
    }
  }

 private:
  /// The body of evaluateJacobian() for `kDim` residuals per element (0: the dimension of `args`). The
  /// arithmetic does not depend on `kDim`; a compile-time dimension lets the compiler unroll the scatter
  /// of an element's rows, which for a dimension known at run time only costs more instructions than
  /// the model itself.
  ///
  /// Kept out of line (the `__restrict__` promise only holds for the parameters of the function being
  /// compiled). The call of the model is forced inline: left to the heuristics, the compiler keeps the
  /// model's code out of the loop and the scatter then reloads everything the model stored.
  template <size_t kDim, class ModelPtr>
  [[gnu::noinline]] static void evaluateJacobianOfDim(ModelPtr models, const KernelArgs<T>& args, size_t first,
                                                      size_t count, const T* /*residuals*/, T* __restrict__ transposed,
                                                      T* __restrict__ jacobian, size_t jacobian_stride) {
    auto& model = *models;
    const size_t params = args.params;
    const size_t dim = kDim != 0 ? kDim : args.observation_dim;

    for (size_t e = 0; e < count; ++e) {
#if defined(__clang__)  // GCC ignores the attribute and warns about it
      [[clang::always_inline]]
#endif
      model.jacobian(args.x, args.input + (first + e) * args.input_dim, args.observations + (first + e) * dim,
                     transposed);

      // The model writes the element's transposed jacobian; the chunk stores one column per parameter.
      T* column = jacobian + e * dim;
      for (size_t p = 0; p < params; ++p, column += jacobian_stride) {
        for (size_t i = 0; i < dim; ++i) {
          column[i] = transposed[p + params * i];
        }
      }
    }
  }
};

}  // namespace detail

/**
 * @brief Cost whose jacobian comes from the model's own `jacobian()`.
 *
 * The model writes the derivative of its residuals with respect to the step `delta` at `delta = 0`
 * (see IModel.hh). With Euclidean parameters, built from `param_dim`, that is the ordinary
 * jacobian with respect to `x`. With a manifold it is the jacobian with respect to the tangent
 * vector of `x [+] delta`, so a pose is differentiated in the convention of the manifold's `plus`.
 *
 * Residuals and jacobian are evaluated in one evaluation per call, through the
 * ExecutionContext (the process-wide default when none is given), possibly on several threads;
 * IModel.hh lists what a model may and may not do.
 */
template <class Model, class T>
  requires AnalyticalModel<Model, T>
class AnalyticalCost : public ChunkedCost<Model, T, detail::AnalyticalJacobian<Model, T>> {
  using Base = ChunkedCost<Model, T, detail::AnalyticalJacobian<Model, T>>;

 public:
  using typename Base::MatrixT;
  using typename Base::VectorT;

  /// Euclidean parameters: `x` has `param_dim` entries. `context` null means defaultContext().
  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 size_t param_dim, Model model = Model{}, size_t active_param_dim = 0,
                 std::shared_ptr<ExecutionContext> context = nullptr)
      : Base(input, observations, num_elements, input_dim, observation_dim, Base::euclideanSpace(param_dim),
             std::move(model), active_param_dim, std::move(context)) {}

  /// Parameters on a manifold. `active_param_dim` counts leading tangent dimensions.
  AnalyticalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                 std::shared_ptr<const IManifold<T>> manifold, Model model = Model{}, size_t active_param_dim = 0,
                 std::shared_ptr<ExecutionContext> context = nullptr)
      : Base(input, observations, num_elements, input_dim, observation_dim, Base::spaceOf(std::move(manifold)),
             std::move(model), active_param_dim, std::move(context)) {}
};

}  // namespace moptim
