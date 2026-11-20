#pragma once
// Eigen-free, device-callable version of the analytical chunk loop of ChunkedCost::computeLinearSystem
// for one cost: residuals of a chunk, jacobian rows, fused SIMD accumulation (copied from
// src/NormalEquations.cc), and a partition-based work-item function whose partials are summed in a
// fixed order by the caller.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>

namespace spike {

using Index = std::ptrdiff_t;

// --- Copied from src/NormalEquations.cc (anonymous namespace there), unchanged except for the
// --- namespace. -------------------------------------------------------------------------------
#if !(defined(__GNUC__) || defined(__clang__))
#error "the fused kernel needs the GCC/Clang vector extension"
#endif

// 128-bit SIMD vectors through the GCC/Clang vector extension: two doubles or four floats.
template <class T>
struct Simd;
template <>
struct Simd<double> {
  using Vector __attribute__((vector_size(16))) = double;
};
template <>
struct Simd<float> {
  using Vector __attribute__((vector_size(16))) = float;
};

// The fused kernel below is only as good as the compiler's ability to keep its accumulators in
// registers, so it is sized by the number of SIMD registers of the target. On AArch64 it was
// measured faster than the dot products up to 16 columns with both GCC and Clang; the limit is
// lower to bound the compile time and code size of the instantiations. The 16-register values are
// a conservative guess that has not been measured.
#if defined(__aarch64__) || defined(__AVX512F__)
constexpr int kSimdRegisters = 32;
constexpr int kMaxFusedCols = 12;
#else
constexpr int kSimdRegisters = 16;
constexpr int kMaxFusedCols = 3;
#endif

/// Accumulators and loaded columns one pass may keep alive, leaving a few registers as scratch.
constexpr int kMaxLiveVectors = kSimdRegisters - 4;

/**
 * Number of columns of the lower triangle of H, starting at column `first`, that one pass over the
 * rows accumulates for a jacobian with `cols` columns.
 *
 * A pass holds one accumulator per entry of those columns and of the matching entries of g, plus
 * the jacobian columns `first..cols-1` and the residuals it loads for each row. It takes as many
 * columns as keep all of that within kMaxLiveVectors, and always at least one.
 */
constexpr int passWidth(int cols, int first) {
  const int loaded = cols - first + 1;
  int accumulators = 0;
  int width = 0;
  while (first + width < cols) {
    const int column_accumulators = cols - (first + width) + 1;
    if (width > 0 && accumulators + column_accumulators + loaded > kMaxLiveVectors) {
      break;
    }
    accumulators += column_accumulators;
    ++width;
  }
  return width;
}

/**
 * Fused kernel for a compile-time number of jacobian columns. One pass down the rows updates every
 * entry of several columns of H (and of g) at once: each row is loaded once per jacobian column
 * instead of once per entry, and the accumulators are independent, so the loop runs at the
 * throughput of the multiply-adds rather than at the latency of one dot product at a time.
 *
 * This instantiation handles the columns of H from `First` on that fit in one pass (passWidth())
 * and hands the rest to the next instantiation. Splitting matters: accumulators that do not fit in
 * registers are spilled to the stack, which costs more than a second pass over rows that are
 * already in the L1 cache and, on the Apple M3 at least, made the run time of the unsplit
 * six-column kernel vary by 45% from one process launch to the next.
 */
template <int Cols, int First, class T>
void accumulateFused(const T* jacobian, Index stride, const T* residuals, Index rows, T* hessian, Index hessian_stride,
                     T* gradient) {
  using Vector = typename Simd<T>::Vector;
  constexpr Index kLanes = sizeof(Vector) / sizeof(T);
  constexpr int kWidth = passWidth(Cols, First);
  constexpr int kEnd = First + kWidth;   // one past the last column of H of this pass
  constexpr int kLoaded = Cols - First;  // jacobian columns First..Cols-1
  constexpr int kPairs = kWidth * kLoaded - kWidth * (kWidth - 1) / 2;

  Vector hessian_lanes[kPairs] = {};
  Vector gradient_lanes[kWidth] = {};

  Index row = 0;
  for (; row + kLanes <= rows; row += kLanes) {
    Vector columns[kLoaded];
    for (int c = 0; c < kLoaded; ++c) {
      std::memcpy(&columns[c], jacobian + (First + c) * stride + row, sizeof(Vector));
    }
    Vector residual;
    std::memcpy(&residual, residuals + row, sizeof(Vector));

    int pair = 0;
    for (int a = First; a < kEnd; ++a) {
      for (int b = a; b < Cols; ++b) {
        hessian_lanes[pair++] += columns[a - First] * columns[b - First];
      }
      gradient_lanes[a - First] += columns[a - First] * residual;
    }
  }

  // Rows that do not fill a vector.
  T hessian_tail[kPairs] = {};
  T gradient_tail[kWidth] = {};
  for (; row < rows; ++row) {
    int pair = 0;
    for (int a = First; a < kEnd; ++a) {
      const T value = jacobian[a * stride + row];
      for (int b = a; b < Cols; ++b) {
        hessian_tail[pair++] += value * jacobian[b * stride + row];
      }
      gradient_tail[a - First] += value * residuals[row];
    }
  }

  int pair = 0;
  for (int a = First; a < kEnd; ++a) {
    for (int b = a; b < Cols; ++b, ++pair) {
      T sum = hessian_tail[pair];
      for (Index lane = 0; lane < kLanes; ++lane) {
        sum += hessian_lanes[pair][lane];
      }
      hessian[b + a * hessian_stride] += sum;
    }
    T sum = gradient_tail[a - First];
    for (Index lane = 0; lane < kLanes; ++lane) {
      sum += gradient_lanes[a - First][lane];
    }
    gradient[a] += sum;
  }

  if constexpr (kEnd < Cols) {
    accumulateFused<Cols, kEnd, T>(jacobian, stride, residuals, rows, hessian, hessian_stride, gradient);
  }
}

/// Calls the kernel instantiated for `cols` columns, one of 1..sizeof...(Is). Returns false for any
/// other column count.
template <class T, size_t... Is>
bool dispatchFused(std::index_sequence<Is...> /*column_counts*/, Index cols, const T* jacobian, Index stride,
                   const T* residuals, Index rows, T* hessian, Index hessian_stride, T* gradient) {
  return ((cols == static_cast<Index>(Is + 1)
               ? (accumulateFused<static_cast<int>(Is + 1), 0, T>(jacobian, stride, residuals, rows, hessian,
                                                                  hessian_stride, gradient),
                  true)
               : false) ||
          ...);
}

// --- Chunk loop ---------------------------------------------------------------------------------

/// Fused accumulation of one chunk: H (lower triangle, column-major, `hessian_stride`) += J^T J and
/// g += J^T r. Out of line, like accumulateNormalEquations() in the library.
[[gnu::noinline]] inline void accumulateNormalEquations(const double* jacobian, Index stride, const double* residuals,
                                                        Index rows, Index cols, double* hessian, Index hessian_stride,
                                                        double* gradient) {
  dispatchFused<double>(std::make_index_sequence<kMaxFusedCols>{}, cols, jacobian, stride, residuals, rows, hessian,
                        hessian_stride, gradient);
}

/// Sum of squares of `n` values with four independent two-lane accumulators, combined in a fixed
/// order. The one reduction used by the host and the kernel path, so both round the same way.
inline double squaredNorm(const double* values, Index n) {
  using Vector = Simd<double>::Vector;
  Vector acc[4] = {};
  Index i = 0;
  for (; i + 8 <= n; i += 8) {
    for (int k = 0; k < 4; ++k) {
      Vector v;
      std::memcpy(&v, values + i + 2 * k, sizeof(Vector));
      acc[k] += v * v;
    }
  }
  Vector sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  double total = sum[0] + sum[1];
  for (; i < n; ++i) total += values[i] * values[i];
  return total;
}

/// Read-only description of one linear system, plus the per-call buffers. Lives in USM; kernels
/// receive a pointer to it.
template <class Model>
struct Problem {
  const Model* model;
  const double* input;
  const double* observations;
  const double* x;
  size_t num_elements;
  size_t input_dim, observation_dim, params;
  size_t chunk_elements;
  size_t chunks_per_partition;
  size_t num_partitions;
  double* scratch;  ///< `scratch_stride` doubles per worker: residuals, jacobian, element jacobian.
  size_t scratch_stride;
  double* partials;  ///< `partial_stride` doubles per partition: packed lower H, g, cost.
  size_t partial_stride;
  size_t* counter;  ///< Next unclaimed partition (kernel mode only).
};

constexpr size_t kMaxParams = kMaxFusedCols;

constexpr size_t packedSize(size_t params) { return params * (params + 1) / 2 + params + 1; }

/// Same sizing as ChunkedCost::chunkElements() with a 16 KiB budget.
inline size_t chunkElements(size_t num_elements, size_t observation_dim, size_t params) {
  const size_t bytes_per_element = observation_dim * (params + 1) * sizeof(double);
  const size_t fitting = (16 * 1024) / std::max<size_t>(bytes_per_element, 1);
  return std::clamp<size_t>(fitting, 1, std::max<size_t>(num_elements, 1));
}

/// The pieces of ChunkedCost / AnalyticalCost that read the model, as out-of-line functions whose
/// output buffers are declared `__restrict__` (see ChunkedCost::evaluateResiduals()).
template <class Model>
struct ChunkEvaluator {
  const Model* model;
  const double* x;
  const double* input;
  const double* observations;
  size_t input_dim, observation_dim, params;

  [[gnu::noinline]] void evaluateResiduals(size_t first, size_t count, double* __restrict__ residuals) const {
    const double* in = input + first * input_dim;
    const double* observation = observations + first * observation_dim;
    for (size_t e = 0; e < count; ++e) {
      model->residual(x, in, observation, residuals);
      in += input_dim;
      observation += observation_dim;
      residuals += observation_dim;
    }
  }

  [[gnu::noinline]] void fillJacobian(size_t first, size_t count, double* __restrict__ transposed,
                                      double* __restrict__ jacobian, size_t jacobian_stride) const {
    const size_t dim = observation_dim;
    for (size_t e = 0; e < count; ++e) {
      [[clang::always_inline]] model->jacobian(x, input + (first + e) * input_dim, observations + (first + e) * observation_dim, transposed);
      for (size_t p = 0; p < params; ++p) {
        double* column = jacobian + p * jacobian_stride + e * dim;
        for (size_t i = 0; i < dim; ++i) {
          column[i] = transposed[p + params * i];
        }
      }
    }
  }
};

/**
 * The work-item function. Claims partitions with `claim()` (returns an index; >= num_partitions
 * means done) and, for each, runs the chunk loop over its whole chunks and writes one packed partial
 * (lower H, g, cost). `worker` selects the scratch slice.
 */
template <class Model, class Claim>
[[gnu::noinline]] void runWorker(const Problem<Model>& problem, size_t worker, Claim claim) {
  const size_t dim = problem.observation_dim;
  const size_t params = problem.params;
  const size_t chunk_rows = problem.chunk_elements * dim;
  const size_t total_chunks = (problem.num_elements + problem.chunk_elements - 1) / problem.chunk_elements;

  double* residuals = problem.scratch + worker * problem.scratch_stride;
  double* jacobian = residuals + chunk_rows;
  double* element_jacobian = jacobian + chunk_rows * params;
  const ChunkEvaluator<Model> evaluator{problem.model,        problem.x,         problem.input,
                                        problem.observations, problem.input_dim, dim,
                                        params};

  for (size_t partition = claim(); partition < problem.num_partitions; partition = claim()) {
    double hessian[kMaxParams * kMaxParams] = {};
    double gradient[kMaxParams] = {};
    double cost = 0.0;

    const size_t first_chunk = partition * problem.chunks_per_partition;
    const size_t last_chunk = std::min(first_chunk + problem.chunks_per_partition, total_chunks);
    for (size_t chunk = first_chunk; chunk < last_chunk; ++chunk) {
      const size_t first = chunk * problem.chunk_elements;
      const size_t count = std::min(problem.chunk_elements, problem.num_elements - first);
      const Index rows = static_cast<Index>(count * dim);

      evaluator.evaluateResiduals(first, count, residuals);
      evaluator.fillJacobian(first, count, element_jacobian, jacobian, chunk_rows);
      cost += squaredNorm(residuals, rows);
      accumulateNormalEquations(jacobian, static_cast<Index>(chunk_rows), residuals, rows, static_cast<Index>(params),
                                hessian, static_cast<Index>(params), gradient);
    }

    double* out = problem.partials + partition * problem.partial_stride;
    for (size_t a = 0; a < params; ++a) {
      for (size_t b = a; b < params; ++b) *out++ = hessian[b + a * params];
    }
    for (size_t a = 0; a < params; ++a) *out++ = gradient[a];
    *out = cost;
  }
}

}  // namespace spike
