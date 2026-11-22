#pragma once

// The work-item function of a chunked cost: no Eigen, no virtual calls, no exceptions, raw pointers
// into USM. A launch runs it on several work-items; each one claims partitions (runs of whole chunks,
// see Partitioning.hh) and runs the chunk loop over them on its own scratch slice:
//
//   residuals of the chunk -> jacobian rows -> whiten / robustify -> J^T J, J^T r and the cost
//
// The result of a partition is one partial (cost, J^T r, J^T J) written in its own slot, so what a
// partition holds does not depend on which work-item computed it.
#include <algorithm>
#include <cstddef>
#include <new>
#include <type_traits>

#include "moptim/LossFunction.hh"
#include "moptim/detail/NormalEquationsKernel.hh"
#include "moptim/detail/Partitioning.hh"

namespace moptim::detail {

/// Byte offsets of the buffers of one work-item's scratch slice. Every offset is a multiple of the
/// cache line (and of the model's alignment), so slices of different work-items never share a line.
struct SliceLayout {
  size_t residuals = 0;
  size_t scales = 0;
  size_t jacobian = 0;
  size_t aux = 0;     ///< Policy scratch (the element's transposed jacobian, backward residuals, ...).
  size_t model = 0;   ///< Room for copies of the models, if the work-items copy them.
  size_t stride = 0;  ///< Size of a slice.
};

inline constexpr size_t kCacheLine = 64;

constexpr size_t roundUp(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

/// Layout of the scratch slice for chunks of `chunk_rows` residual rows with `params` jacobian
/// columns, `aux_values` values of policy scratch and `model_bytes` of model copies. The slice must
/// start at a multiple of `sliceAlignment(model_align)`.
constexpr SliceLayout makeSliceLayout(size_t value_bytes, size_t chunk_rows, size_t params, size_t aux_values,
                                      size_t model_bytes, size_t model_align) {
  SliceLayout layout;
  size_t offset = 0;
  layout.residuals = offset;
  offset += roundUp(chunk_rows * value_bytes, kCacheLine);
  layout.scales = offset;
  offset += roundUp(chunk_rows * value_bytes, kCacheLine);
  layout.jacobian = offset;
  offset += roundUp(chunk_rows * params * value_bytes, kCacheLine);
  layout.aux = offset;
  offset += roundUp(aux_values * value_bytes, kCacheLine);
  layout.model = roundUp(offset, std::max(model_align, kCacheLine));
  offset = layout.model + roundUp(model_bytes, kCacheLine);
  layout.stride = roundUp(offset, std::max(model_align, kCacheLine));
  return layout;
}

constexpr size_t sliceAlignment(size_t model_align) { return std::max(model_align, kCacheLine); }

/// Everything a launch reads or writes, as plain data (pointers into USM). It is refreshed by the
/// host before each launch and read by every work-item.
template <class T>
struct KernelArgs {
  // The problem.
  const T* x = nullptr;
  const T* input = nullptr;
  const T* observations = nullptr;
  size_t num_elements = 0;
  size_t input_dim = 0;
  size_t observation_dim = 0;
  size_t params = 0;  ///< Active parameters: columns of the jacobian.
  size_t param_dim = 0;  ///< Entries of `x`.
  Partitioning partitioning;

  // Finite differences (numerical policies): evaluation point `q` (0-based, one row of `param_dim`
  // entries) of the jacobian; `inv_span` is 1 / the distance between the points of a parameter.
  const T* points = nullptr;
  T inv_span = T{0};
  size_t model_count = 1;  ///< Models in the array the work-items use (1 + points).

  // Noise model and robust kernel. Null / false when not set.
  const T* weights = nullptr;                ///< One weight per element.
  const T* information_factors = nullptr;    ///< Per element, the column-major lower Cholesky factor (O x O).
  LossDescriptor<T> loss;
  bool has_loss = false;

  // Buffers of the launch.
  unsigned char* scratch = nullptr;  ///< One slice per work-item.
  SliceLayout slice;
  T* partials = nullptr;             ///< One slot per partition: cost, J^T r (params), J^T J (params x params).
  size_t partial_stride = 0;         ///< Entries of a slot.
  size_t* counter = nullptr;         ///< Next unclaimed partition.

  // WorkGroup kernel (see WorkGroupKernel.hh).
  size_t local_size = 0;             ///< Work-items (elements) per group.
  size_t group_count = 0;
  size_t aux_per_element = 0;        ///< Values of policy scratch per element, in local memory.
  T* group_partials = nullptr;       ///< One slot of `partial_stride` entries per group.
  T* result = nullptr;               ///< The sum over the groups, one slot.
  unsigned int* group_counter = nullptr;  ///< Groups that finished; the last one reduces.
};

/// Entries of a partial slot: the cost, the gradient and the dense `params x params` hessian (only
/// its lower triangle is written), rounded up to whole cache lines.
constexpr size_t partialStride(size_t value_bytes, size_t params) {
  return roundUp(1 + params + params * params, kCacheLine / value_bytes);
}

/// How the work-items reach the model.
enum class ModelAccess {
  Shared,             ///< Every work-item calls the one model through a const reference.
  PerWorkItemCopy,    ///< The member functions are not const: each work-item copies the model into its slice.
  Exclusive,          ///< Not const-callable and not copyable: one work-item uses the model in place.
};

template <class Policy, class Model>
constexpr ModelAccess modelAccess() {
  if constexpr (Policy::kConstCallable) {
    return ModelAccess::Shared;
  } else if constexpr (std::is_copy_constructible_v<Model>) {
    return ModelAccess::PerWorkItemCopy;
  } else {
    return ModelAccess::Exclusive;
  }
}

/// Evaluates the residuals of elements [first, first + count) into `residuals`. `residuals` is a
/// buffer of the work-item that nothing else points into; declaring it so lets the compiler keep the
/// model's state in registers across the loop instead of reloading it after every residual is
/// stored. The function is kept out of line because GCC (and clang, from a lambda) only honour that
/// promise for the parameters of the function being compiled, not once the loop was inlined.
template <class T, class ModelRef>
[[gnu::noinline]] void evaluateResiduals(ModelRef& model, const T* x, const KernelArgs<T>& args, size_t first,
                                         size_t count, T* __restrict__ residuals) {
  const T* input = args.input + first * args.input_dim;
  const T* observation = args.observations + first * args.observation_dim;
  for (size_t e = 0; e < count; ++e) {
    model.residual(x, input, observation, residuals);
    input += args.input_dim;
    observation += args.observation_dim;
    residuals += args.observation_dim;
  }
}

/// column <- L^T * column, in place, for the `dim` entries of one element. `factor` is the
/// column-major lower factor L. Row i of L^T only reads entries >= i, so going down the rows never
/// reads an entry that was already overwritten.
template <class T>
inline void whiten(const T* factor, size_t dim, T* column) {
  for (size_t row = 0; row < dim; ++row) {
    T sum{0};
    for (size_t k = row; k < dim; ++k) {
      sum += factor[k + row * dim] * column[k];
    }
    column[row] = sum;
  }
}

/**
 * The body of robustifyChunk() for elements of `kDim` residuals (0: the dimension of `args`).
 * The arithmetic does not depend on `kDim`: it only lets the compiler unroll the loops over the
 * entries of an element, which it does not do for a dimension known at run time only.
 */
template <size_t kDim, class T>
[[gnu::noinline]] T robustifyChunkOfDim(const KernelArgs<T>& args, size_t first, size_t count,
                                        T* __restrict__ residuals, T* __restrict__ jacobian, size_t jacobian_stride,
                                        size_t cols, T* __restrict__ scales) {
  const size_t dim = kDim != 0 ? kDim : args.observation_dim;
  const bool has_information = args.information_factors != nullptr;
  const bool has_weights = args.weights != nullptr;

  T cost{0};
  for (size_t e = 0; e < count; ++e) {
    T* residual = residuals + e * dim;
    if (has_information) {
      const T* factor = args.information_factors + (first + e) * dim * dim;
      whiten(factor, dim, residual);
      for (size_t col = 0; col < cols; ++col) {
        whiten(factor, dim, jacobian + col * jacobian_stride + e * dim);
      }
    }

    T squared_norm{0};
    for (size_t i = 0; i < dim; ++i) {
      squared_norm += residual[i] * residual[i];
    }

    T scale{1};
    if (has_weights) {
      const T weight = args.weights[first + e];
      squared_norm *= weight;
      scale = lossSqrt(weight);
    }
    if (args.has_loss) {
      T rho[3];
      evaluateLoss(args.loss, squared_norm, rho);
      cost += rho[0];
      scale *= lossSqrt(rho[1]);
    } else {
      cost += squared_norm;
    }
    for (size_t i = 0; i < dim; ++i) {
      scales[e * dim + i] = scale;
    }
  }

  // The scales of all the rows of the chunk are applied at once, one vectorizable pass per column.
  if (cols > 0 && (has_weights || args.has_loss)) {
    const size_t rows = count * dim;
    for (size_t i = 0; i < rows; ++i) {
      residuals[i] *= scales[i];
    }
    for (size_t col = 0; col < cols; ++col) {
      T* column = jacobian + col * jacobian_stride;
      for (size_t i = 0; i < rows; ++i) {
        column[i] *= scales[i];
      }
    }
  }
  return cost;
}

/**
 * Applies the noise model and the robust kernel to a chunk in place and returns its cost.
 *
 * With W_e the whitening of element e (sqrt of its weight, or L_e^T for an information matrix
 * L_e L_e^T) and s_e = ||W_e r_e||^2, the element costs rho(s_e), and its residual block and jacobian
 * rows become sqrt(rho'(s_e)) * W_e * (r_e, J_e): the first-order (IRLS) correction. Without a kernel
 * rho(s) = s.
 *
 * `cols` = 0 computes only the cost (the residuals are left whitened but unscaled). The linear-system
 * and the cost-only launches both call this one function, so the cost of a chunk is bit-identical in
 * the two.
 */
template <class T>
T robustifyChunk(const KernelArgs<T>& args, size_t first, size_t count, T* residuals, T* jacobian,
                 size_t jacobian_stride, size_t cols, T* scales) {
  switch (args.observation_dim) {
    case 1:
      return robustifyChunkOfDim<1>(args, first, count, residuals, jacobian, jacobian_stride, cols, scales);
    case 2:
      return robustifyChunkOfDim<2>(args, first, count, residuals, jacobian, jacobian_stride, cols, scales);
    case 3:
      return robustifyChunkOfDim<3>(args, first, count, residuals, jacobian, jacobian_stride, cols, scales);
    case 4:
      return robustifyChunkOfDim<4>(args, first, count, residuals, jacobian, jacobian_stride, cols, scales);
    default:
      return robustifyChunkOfDim<0>(args, first, count, residuals, jacobian, jacobian_stride, cols, scales);
  }
}

/// Fused accumulation of one chunk, out of line so the kernel stays small and every chunk loop calls
/// the same code: H (lower triangle) += J^T J and g += J^T r.
template <class T>
[[gnu::noinline]] void accumulateChunk(const T* jacobian, size_t jacobian_stride, const T* residuals, size_t rows,
                                       size_t cols, T* hessian, T* gradient) {
  accumulateNormalEquationsRaw<T>(jacobian, static_cast<Index>(jacobian_stride), residuals, static_cast<Index>(rows),
                                  static_cast<Index>(cols), hessian, static_cast<Index>(cols), gradient);
}

/// The chunk loop over the partitions this work-item claims. `ModelRef` is `const Model&` or `Model&`.
template <class Policy, bool kLinearSystem, class T, class ModelPtr, class Claim>
void runPartitions(const KernelArgs<T>& args, ModelPtr models, T* residuals, T* scales, T* jacobian, T* aux,
                   Claim& claim) {
  const Partitioning& parts = args.partitioning;
  const size_t dim = args.observation_dim;
  const size_t params = args.params;
  const size_t chunk_rows = parts.chunk_elements * dim;
  const bool robust = args.has_loss || args.weights != nullptr || args.information_factors != nullptr;

  for (size_t partition = claim(); partition < parts.num_partitions; partition = claim()) {
    T* partial = args.partials + partition * args.partial_stride;
    T* gradient = partial + 1;
    T* hessian = partial + 1 + params;
    if constexpr (kLinearSystem) {
      std::fill_n(gradient, params + params * params, T{0});
    }

    T cost{0};
    const size_t first_chunk = partition * parts.chunks_per_partition;
    const size_t last_chunk = std::min(first_chunk + parts.chunks_per_partition, parts.num_chunks);
    for (size_t chunk = first_chunk; chunk < last_chunk; ++chunk) {
      const size_t first = chunk * parts.chunk_elements;
      const size_t count = std::min(parts.chunk_elements, args.num_elements - first);
      const size_t rows = count * dim;

      evaluateResiduals(*models, args.x, args, first, count, residuals);
      if constexpr (kLinearSystem) {
        Policy::evaluateJacobian(models, args, first, count, residuals, aux, jacobian, chunk_rows);
      }
      cost += robust ? robustifyChunk(args, first, count, residuals, jacobian, chunk_rows,
                                      kLinearSystem ? params : size_t{0}, scales)
                     : squaredNorm(residuals, rows);
      if constexpr (kLinearSystem) {
        accumulateChunk(jacobian, chunk_rows, residuals, rows, params, hessian, gradient);
      }
    }
    partial[0] = cost;
  }
}

/**
 * The evaluation-point models of a numerical policy as one model moved from point to point, for a
 * single work-item that visits each point once (all the elements in one chunk): `models[i]` sets the
 * model's state to point `i` (0 is `x`, `i > 0` is row `i - 1` of `points`) when it is not there
 * already. Host code only, since `setState()` runs inside the walk.
 */
template <class Model, class T>
struct SteppedModels {
  Model* model;
  const T* x;
  const T* points;
  size_t param_dim;
  size_t* current;  ///< The point the model's state is at, shared by the copies of this handle.

  Model& operator[](size_t point) const {
    if (*current != point) {
      model->setState(point == 0 ? x : points + (point - 1) * param_dim);
      *current = point;
    }
    return *model;
  }
  Model& operator*() const { return (*this)[0]; }
};

/// The work-item function over models used in place: `models` is a model pointer or SteppedModels.
template <class Policy, bool kLinearSystem, class T, class ModelPtr, class Claim>
void runWorkItemInPlace(const KernelArgs<T>& args, ModelPtr models, size_t worker, Claim& claim) {
  unsigned char* slice = args.scratch + worker * args.slice.stride;
  T* residuals = reinterpret_cast<T*>(slice + args.slice.residuals);
  T* scales = reinterpret_cast<T*>(slice + args.slice.scales);
  T* jacobian = reinterpret_cast<T*>(slice + args.slice.jacobian);
  T* aux = reinterpret_cast<T*>(slice + args.slice.aux);
  runPartitions<Policy, kLinearSystem>(args, models, residuals, scales, jacobian, aux, claim);
}

/**
 * The function every work-item of a launch runs. `claim()` returns the next unclaimed partition
 * (a value >= the number of partitions means there is none left); `worker` selects the scratch slice.
 *
 * `models` is the array of `args.model_count` models (the first is the one at `x`; a numerical policy
 * has one more per evaluation point). `Policy` supplies the jacobian rows (`kConstCallable` and
 * `evaluateJacobian()`, see AnalyticalCost.hh and NumericalCost.hh). With ModelAccess::PerWorkItemCopy the
 * work-item copy-constructs the models it needs (all of them for a linear system, the first for the cost
 * alone) into its slice once, and destroys the copies before returning; otherwise it uses the shared
 * models in place.
 */
template <class Policy, class Model, class T, bool kLinearSystem, class Claim>
[[gnu::noinline]] void runWorkItem(const KernelArgs<T>& args, Model* models, size_t worker, Claim claim) {
  constexpr ModelAccess kAccess = modelAccess<Policy, Model>();
  if constexpr (kAccess == ModelAccess::Shared) {
    const Model* shared = models;
    runWorkItemInPlace<Policy, kLinearSystem>(args, shared, worker, claim);
  } else if constexpr (kAccess == ModelAccess::PerWorkItemCopy) {
    const size_t used = kLinearSystem ? args.model_count : size_t{1};
    Model* local = reinterpret_cast<Model*>(args.scratch + worker * args.slice.stride + args.slice.model);
    for (size_t i = 0; i < used; ++i) {
      ::new (static_cast<void*>(local + i)) Model(static_cast<const Model&>(models[i]));
    }
    runWorkItemInPlace<Policy, kLinearSystem>(args, local, worker, claim);
    for (size_t i = 0; i < used; ++i) {
      local[i].~Model();
    }
  } else {
    runWorkItemInPlace<Policy, kLinearSystem>(args, models, worker, claim);
  }
}

}  // namespace moptim::detail
