#pragma once

// Normal-equation accumulation routines: no Eigen, no exceptions, no virtuals, raw pointers with explicit strides
// (see src/NormalEquations.cc for the Eigen entry point).
//
// Everything is `H += J^T J` (lower triangle only) and `g += J^T r`, with J column-major.
#include <cstddef>
#include <cstring>
#include <utility>

#if !(defined(__GNUC__) || defined(__clang__))
#error "NormalEquationsKernel.hh needs the GCC/Clang vector extension"
#endif

namespace moptim::detail {

using Index = std::ptrdiff_t;

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

// The fused routine below is only as good as the compiler's ability to keep its accumulators in
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
 * Fused accumulation for a compile-time number of jacobian columns. One pass down the rows updates every
 * entry of several columns of H (and of g) at once: each row is loaded once per jacobian column
 * instead of once per entry, and the accumulators are independent, so the loop runs at the
 * throughput of the multiply-adds rather than at the latency of one dot product at a time.
 *
 * This instantiation handles the columns of H from `First` on that fit in one pass (passWidth())
 * and hands the rest to the next instantiation. Splitting matters: accumulators that do not fit in
 * registers are spilled to the stack, which costs more than a second pass over rows that are
 * already in the L1 cache and, on the Apple M3 at least, made the run time of the unsplit
 * six-column version vary by 45% from one process start to the next.
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

/// Calls the fused routine instantiated for `cols` columns, one of 1..sizeof...(Is). Returns false for any
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

/**
 * Sum over `n` entries of `x[i] * y[i]` with four independent SIMD accumulators combined in a fixed
 * order, so the result depends only on `n` and the data, wherever it is called from. Used
 * where the fused routine does not apply (more columns than kMaxFusedCols).
 */
template <class T>
T dot(const T* x, const T* y, Index n) {
  using Vector = typename Simd<T>::Vector;
  constexpr Index kLanes = sizeof(Vector) / sizeof(T);
  Vector acc[4] = {};
  Index i = 0;
  for (; i + 4 * kLanes <= n; i += 4 * kLanes) {
    for (int k = 0; k < 4; ++k) {
      Vector a;
      Vector b;
      std::memcpy(&a, x + i + k * kLanes, sizeof(Vector));
      std::memcpy(&b, y + i + k * kLanes, sizeof(Vector));
      acc[k] += a * b;
    }
  }
  const Vector sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  T total = sum[0];
  for (Index lane = 1; lane < kLanes; ++lane) {
    total += sum[lane];
  }
  for (; i < n; ++i) {
    total += x[i] * y[i];
  }
  return total;
}

/// Sum of squares of `n` values: the one fixed-order SIMD reduction of the squared norm of a residual
/// block, shared by all callers so that they round the same way.
template <class T>
T squaredNorm(const T* values, size_t n) {
  return dot(values, values, static_cast<Index>(n));
}

/// One dot product per entry of the lower triangle. Works for any column count.
template <class T>
void accumulateByDotProducts(const T* jacobian, Index stride, const T* residuals, Index rows, Index cols, T* hessian,
                             Index hessian_stride, T* gradient) {
  for (Index a = 0; a < cols; ++a) {
    const T* column = jacobian + a * stride;
    for (Index b = a; b < cols; ++b) {
      hessian[b + a * hessian_stride] += dot(column, jacobian + b * stride, rows);
    }
    gradient[a] += dot(column, residuals, rows);
  }
}

/**
 * `H += J^T J` (lower triangle of the `cols` x `cols` matrix H, column stride `hessian_stride`) and
 * `g += J^T r` for the column-major `rows` x `cols` jacobian with column stride `stride`. The fused
 * fused routine handles up to kMaxFusedCols columns; more columns take dot products, whose rounding differs
 * from the Eigen dot products this fallback used before the fused routine.
 */
template <class T>
void accumulateNormalEquationsRaw(const T* jacobian, Index stride, const T* residuals, Index rows, Index cols,
                                  T* hessian, Index hessian_stride, T* gradient) {
  const bool fused = dispatchFused<T>(std::make_index_sequence<kMaxFusedCols>{}, cols, jacobian, stride, residuals,
                                      rows, hessian, hessian_stride, gradient);
  if (!fused) {
    accumulateByDotProducts(jacobian, stride, residuals, rows, cols, hessian, hessian_stride, gradient);
  }
}

}  // namespace moptim::detail
