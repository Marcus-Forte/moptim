#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "moptim/ChunkedCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/IModel.hh"
#include "moptim/detail/ChunkKernel.hh"

namespace moptim {

/// Finite-difference formula used by NumericalCost.
enum class DifferenceScheme {
  Forward,  ///< (r(x + h) - r(x)) / h: one extra evaluation per parameter, O(h) truncation error.
  Central,  ///< (r(x + h) - r(x - h)) / 2h: two per parameter, O(h^2) truncation error.
};

namespace detail {

/**
 * Jacobian policy of NumericalCost: the columns of a chunk's jacobian come from residuals at
 * evaluation points. Point 0 is `x`; each active parameter `p` has `kPointsPerParam` more, at
 * index `1 + kPointsPerParam * p` (`x` stepped up by `h`) and, in the central scheme, the next one
 * (stepped down). Every point has its own model, whose state the host set to that point.
 */
template <class Model, class T, DifferenceScheme Scheme>
struct NumericalJacobian {
  /// Only `residual` is called, so work-items can share the models if it is const-callable.
  static constexpr bool kConstCallable = ConstResidualModel<Model, T>;

  static constexpr size_t kPointsPerParam = Scheme == DifferenceScheme::Central ? 2 : 1;

  /// Finite-difference step: sqrt of the machine epsilon.
  static T step() { return std::sqrt(std::numeric_limits<T>::epsilon()); }

  /// 1 / the distance between the evaluation points of a parameter.
  static T invSpan() { return T{1} / (static_cast<T>(kPointsPerParam) * step()); }

  /// Values of scratch: the central scheme keeps the backward residuals of a chunk.
  static constexpr size_t auxValues(size_t chunk_rows, size_t /*params*/, size_t /*observation_dim*/) {
    return Scheme == DifferenceScheme::Central ? chunk_rows : 0;
  }

  /// Fills each column of the chunk's jacobian straight from the residuals at its evaluation points.
  /// `residuals` are those of the chunk at `x`. The buffers are the work-item's own and nothing else
  /// points into them (see evaluateResiduals() for why they are declared so and why this is not inlined).
  template <class ModelPtr>
  [[gnu::noinline]] static void evaluateJacobian(ModelPtr models, const KernelArgs<T>& args, size_t first,
                                                 size_t count, const T* __restrict__ residuals,
                                                 T* __restrict__ backward, T* __restrict__ jacobian,
                                                 size_t jacobian_stride) {
    const size_t rows = count * args.observation_dim;
    const T inv_span = args.inv_span;
    for (size_t p = 0; p < args.params; ++p) {
      T* column = jacobian + p * jacobian_stride;
      const size_t point = kPointsPerParam * p;
      evaluateResiduals(models[1 + point], args.points + point * args.param_dim, args, first, count, column);

      if constexpr (Scheme == DifferenceScheme::Central) {
        evaluateResiduals(models[2 + point], args.points + (point + 1) * args.param_dim, args, first, count,
                          backward);
        for (size_t i = 0; i < rows; ++i) {
          column[i] = (column[i] - backward[i]) * inv_span;
        }
      } else {
        for (size_t i = 0; i < rows; ++i) {
          column[i] = (column[i] - residuals[i]) * inv_span;
        }
      }
    }
  }
};

}  // namespace detail

/**
 * @brief Cost whose jacobian is obtained by finite differences of the model's residuals.
 *
 * The residuals are evaluated at one or two extra points per active parameter: `x` with that
 * parameter stepped up, or up and down. A chunk of elements is run through all of these points
 * while its inputs and observations are still in cache, instead of making one pass over the whole
 * data set per parameter.
 *
 * Every evaluation point has its own copy of the model, with its state set to the point before each
 * launch: `1 + P_active` models for the forward scheme and `1 + 2 P_active` for the central one, all
 * copy-constructed from the model given, once, on the first evaluation that needs them (see
 * ChunkedCost). A cost whose elements fit in one chunk, with a const-callable
 * model, needs none: its one model goes from point to point. Copies must not share state that
 * `setState()` or `residual()` write to. Everything else (kernel, partitions,
 * noise model, determinism) is ChunkedCost's; `residual` may run on several threads (IModel.hh).
 *
 * On a manifold the evaluation points are `x [+] (+h e_p)` and `x [+] (-h e_p)` for each active
 * tangent dimension `p`, so the jacobian is the derivative with respect to the step at zero, the
 * same quantity an analytical model provides. The points are computed on the host once per linear
 * system. Without a manifold they are `x` with entry `p` stepped in place.
 *
 * Use it through the NumericalCostForwardEuler and NumericalCostCentral aliases.
 */
template <class Model, class T, DifferenceScheme Scheme>
  requires NumericalModel<Model, T>
class NumericalCost : public ChunkedCost<Model, T, detail::NumericalJacobian<Model, T, Scheme>> {
  using Base = ChunkedCost<Model, T, detail::NumericalJacobian<Model, T, Scheme>>;

 public:
  using typename Base::MatrixT;
  using typename Base::VectorT;

  /// Euclidean parameters: `x` has `param_dim` entries. `context` null means defaultContext().
  NumericalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                size_t param_dim, Model model = Model{}, size_t active_param_dim = 0,
                std::shared_ptr<ExecutionContext> context = nullptr)
      : Base(input, observations, num_elements, input_dim, observation_dim, Base::euclideanSpace(param_dim),
             std::move(model), active_param_dim, std::move(context)) {}

  /// Parameters on a manifold. `active_param_dim` counts leading tangent dimensions.
  NumericalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                std::shared_ptr<const IManifold<T>> manifold, Model model = Model{}, size_t active_param_dim = 0,
                std::shared_ptr<ExecutionContext> context = nullptr)
      : Base(input, observations, num_elements, input_dim, observation_dim, Base::spaceOf(std::move(manifold)),
             std::move(model), active_param_dim, std::move(context)) {}
};

}  // namespace moptim
