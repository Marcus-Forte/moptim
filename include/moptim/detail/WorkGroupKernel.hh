#pragma once

// The work-group kernel of a cost: one element per work-item, for GPUs and other accelerators.
//
// A launch has ceil(N / L) work-groups of L work-items (the last one may have fewer elements). Group g
// owns the elements [g * L, min((g + 1) * L, N)), a range that depends on nothing but N and L, so what
// a group computes does not depend on scheduling. Each work-item
//
//   evaluates the residuals and jacobian rows of its element into work-group local memory
//   -> whitens / robustifies them
//
// and, after a barrier, the group's work-items compute the group's sums together: the cost, J^T r and the
// lower triangle of J^T J over the group's rows, entry k by work-item k, k + L, ... Each entry is one
// sequential sum over the rows in order. The group writes them as its partial. The last group to
// finish (counted with an atomic, fenced) adds the partials of all the groups, in group order, into
// the result, so the host reads one slot, not one per group.
//
// The device code uses 32-bit atomics only, no Eigen, no virtual calls, no function pointers and the
// SYCL math functions (see lossSqrt in LossFunction.hh).
#include <algorithm>
#include <cstddef>
#include <sycl/sycl.hpp>

#include "moptim/detail/ChunkKernel.hh"

namespace moptim::detail {

/// Where the regions of the work-group's local memory start, in values of T.
struct WorkGroupLayout {
  size_t residuals = 0;  ///< L * O: the residuals of the group's elements, element after element.
  size_t jacobian = 0;   ///< P columns of L * O rows (a linear system only).
  size_t aux = 0;        ///< L * aux_per_element (a linear system only).
  size_t costs = 0;      ///< L: the cost of each element.
  size_t total = 0;
};

constexpr WorkGroupLayout makeWorkGroupLayout(size_t local_size, size_t observation_dim, size_t params,
                                              size_t aux_per_element, bool linear_system) {
  WorkGroupLayout layout;
  const size_t rows = local_size * observation_dim;
  size_t offset = rows;
  layout.jacobian = offset;
  if (linear_system) {
    offset += rows * params;
  }
  layout.aux = offset;
  if (linear_system) {
    offset += local_size * aux_per_element;
  }
  layout.costs = offset;
  layout.total = offset + local_size;
  return layout;
}

/// Local-memory values one element needs in a linear system.
constexpr size_t workGroupValuesPerElement(size_t observation_dim, size_t params, size_t aux_per_element) {
  return observation_dim * (1 + params) + aux_per_element + 1;
}

/// Entries of a group's partial: the cost, the gradient and the lower triangle of the hessian.
constexpr size_t workGroupEntries(size_t params, bool linear_system) {
  return linear_system ? 1 + params + params * (params + 1) / 2 : 1;
}

/// Where entry `k` of a group's partial is stored in the slot (see ChunkedCost: cost, then the
/// gradient, then the dense params x params hessian, of which the lower triangle is written).
/// `a` and `b` are the row and column of a hessian entry, for the sums.
struct WorkGroupEntry {
  size_t offset;
  int kind;  ///< 0 cost, 1 gradient entry `a`, 2 hessian entry (b, a), b >= a.
  size_t a;
  size_t b;
};

inline WorkGroupEntry workGroupEntry(size_t k, size_t params) {
  if (k == 0) {
    return {0, 0, 0, 0};
  }
  if (k <= params) {
    return {k, 1, k - 1, 0};
  }
  size_t pair = k - 1 - params;
  size_t a = 0;
  while (pair >= params - a) {
    pair -= params - a;
    ++a;
  }
  const size_t b = a + pair;
  return {1 + params + b + a * params, 2, a, b};
}

/// Applies the noise model and the robust kernel to one element in place and returns its cost: the
/// same arithmetic as robustifyChunk() for a chunk of one element (the residual is whitened, its squared
/// norm weighted and passed through the loss, and the residual and jacobian rows scaled by
/// sqrt(rho') times the weight's root). With `cols` = 0 only the cost is formed.
template <class T>
T robustifyElement(const KernelArgs<T>& args, size_t element, T* residual, T* jacobian, size_t jacobian_stride,
                   size_t cols) {
  const size_t dim = args.observation_dim;
  if (args.information_factors != nullptr) {
    const T* factor = args.information_factors + element * dim * dim;
    whiten(factor, dim, residual);
    for (size_t col = 0; col < cols; ++col) {
      whiten(factor, dim, jacobian + col * jacobian_stride);
    }
  }

  T squared_norm{0};
  for (size_t i = 0; i < dim; ++i) {
    squared_norm += residual[i] * residual[i];
  }

  T scale{1};
  if (args.weights != nullptr) {
    const T weight = args.weights[element];
    squared_norm *= weight;
    scale = lossSqrt(weight);
  }
  T cost = squared_norm;
  if (args.has_loss) {
    T rho[3];
    evaluateLoss(args.loss, squared_norm, rho);
    cost = rho[0];
    scale *= lossSqrt(rho[1]);
  }

  if (cols > 0 && (args.weights != nullptr || args.has_loss)) {
    for (size_t i = 0; i < dim; ++i) {
      residual[i] *= scale;
    }
    for (size_t col = 0; col < cols; ++col) {
      for (size_t i = 0; i < dim; ++i) {
        jacobian[col * jacobian_stride + i] *= scale;
      }
    }
  }
  return cost;
}

/// The group's sums: entry k of the partial by work-item k % L, each one a sequential sum over the
/// group's rows in order. Out of line: the DPC++ native_cpu vectorizer crashes on a kernel that has
/// these loops and the final reduction's inlined into it.
template <class T>
[[gnu::noinline]] void workGroupSums(const KernelArgs<T>& args, size_t t, size_t group, size_t count,
                                     const T* residuals, const T* jacobian, const T* costs, size_t row_stride,
                                     bool linear_system) {
  const size_t local_size = args.local_size;
  const size_t params = args.params;
  const size_t rows = count * args.observation_dim;
  const size_t entries = workGroupEntries(params, linear_system);
  T* partial = args.group_partials + group * args.partial_stride;
  for (size_t k = t; k < entries; k += local_size) {
    const WorkGroupEntry entry = workGroupEntry(k, params);
    T sum{0};
    if (entry.kind == 0) {
      for (size_t e = 0; e < count; ++e) {
        sum += costs[e];
      }
    } else if (entry.kind == 1) {
      const T* column = jacobian + entry.a * row_stride;
      for (size_t r = 0; r < rows; ++r) {
        sum += column[r] * residuals[r];
      }
    } else {
      const T* column_a = jacobian + entry.a * row_stride;
      const T* column_b = jacobian + entry.b * row_stride;
      for (size_t r = 0; r < rows; ++r) {
        sum += column_a[r] * column_b[r];
      }
    }
    partial[entry.offset] = sum;
  }
}

/// The last group's sum of the groups' partials, in group order, into the result.
template <class T>
[[gnu::noinline]] void workGroupFinalSums(const KernelArgs<T>& args, size_t t, bool linear_system) {
  const size_t entries = workGroupEntries(args.params, linear_system);
  for (size_t k = t; k < entries; k += args.local_size) {
    const WorkGroupEntry entry = workGroupEntry(k, args.params);
    T sum{0};
    for (size_t g = 0; g < args.group_count; ++g) {
      sum += args.group_partials[g * args.partial_stride + entry.offset];
    }
    args.result[entry.offset] = sum;
  }
}

/**
 * The function every work-item of a work-group launch runs. `models` is the array of
 * `args.model_count` models, called through const references (the cost only uses this kernel for models
 * whose member functions are const). `local` and `last_flag` are the group's local memory.
 */
template <class Policy, class T, class Model, bool kLinearSystem>
void runWorkGroupItem(const KernelArgs<T>& args, const Model* models, sycl::nd_item<1> item, T* local,
                      unsigned int* last_flag) {
  const size_t t = item.get_local_id(0);
  const size_t group = item.get_group(0);
  const size_t local_size = args.local_size;
  const size_t dim = args.observation_dim;
  const size_t first = group * local_size;
  const size_t count = std::min(local_size, args.num_elements - first);
  const size_t row_stride = local_size * dim;

  const WorkGroupLayout layout =
      makeWorkGroupLayout(local_size, dim, args.params, args.aux_per_element, kLinearSystem);
  T* residuals = local + layout.residuals;
  T* jacobian = local + layout.jacobian;
  T* aux = local + layout.aux;
  T* costs = local + layout.costs;

  if (t < count) {
    const size_t element = first + t;
    T* residual = residuals + t * dim;
    models[0].residual(args.x, args.input + element * args.input_dim, args.observations + element * dim, residual);
    T* element_jacobian = jacobian + t * dim;
    if constexpr (kLinearSystem) {
      Policy::evaluateElementJacobian(models, args, element, residual, aux + t * args.aux_per_element,
                                      element_jacobian, row_stride);
    }
    costs[t] =
        robustifyElement(args, element, residual, element_jacobian, row_stride, kLinearSystem ? args.params : 0);
  }
  sycl::group_barrier(item.get_group());

  workGroupSums(args, t, group, count, residuals, jacobian, costs, row_stride, kLinearSystem);

  // The last group to get here adds the partials of all the groups, in group order. Ordering:
  //  1. The partial's entries are written by many work-items. This barrier, with device-scope memory
  //     ordering, makes every one of those writes happen before work-item 0's atomic below.
  //  2. Work-item 0 increments the counter with acq_rel at device scope: its release half publishes this
  //     group's partial to whichever group counts last, and, in the group that does count last, its acquire
  //     half makes the other groups' (released) partials visible to work-item 0.
  //  3. The second barrier orders every other work-item of the group after work-item 0's acquire, so by
  //     the transitivity of happens-before their reads of the partials see the other groups' writes. That is
  //     why they need no acquire of their own.
  sycl::group_barrier(item.get_group(), sycl::memory_scope::device);
  if (t == 0) {
    sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel, sycl::memory_scope::device> finished(
        *args.group_counter);
    *last_flag = finished.fetch_add(1u) + 1u == args.group_count ? 1u : 0u;
  }
  sycl::group_barrier(item.get_group(), sycl::memory_scope::device);
  if (*last_flag != 0u) {
    workGroupFinalSums(args, t, kLinearSystem);
  }
}

}  // namespace moptim::detail
