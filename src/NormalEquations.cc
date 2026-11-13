#include "moptim/NormalEquations.hh"

#include <cassert>
#include <cstring>
#include <utility>

namespace moptim {
namespace {

using Index = Eigen::Index;

template <class T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T>
using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/// One dot product per entry. Works for any size; Eigen vectorizes each dot product along the rows.
template <class T>
void accumulateByDotProducts(const Eigen::Ref<const MatrixT<T>>& jacobian,
                             const Eigen::Ref<const VectorT<T>>& residuals, Eigen::Ref<MatrixT<T>> hessian,
                             Eigen::Ref<VectorT<T>> gradient) {
  for (Index a = 0; a < jacobian.cols(); ++a) {
    const auto column = jacobian.col(a);
    for (Index b = a; b < jacobian.cols(); ++b) {
      hessian(b, a) += column.dot(jacobian.col(b));
    }
    gradient(a) += column.dot(residuals);
  }
}

#if defined(__GNUC__) || defined(__clang__)

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

/// Runs the fused kernel when there is one for this column count. Returns false otherwise.
template <class T>
bool accumulateFusedIfAvailable(const Eigen::Ref<const MatrixT<T>>& jacobian,
                                const Eigen::Ref<const VectorT<T>>& residuals, Eigen::Ref<MatrixT<T>> hessian,
                                Eigen::Ref<VectorT<T>> gradient) {
  return dispatchFused<T>(std::make_index_sequence<kMaxFusedCols>{}, jacobian.cols(), jacobian.data(),
                          jacobian.outerStride(), residuals.data(), jacobian.rows(), hessian.data(),
                          hessian.outerStride(), gradient.data());
}

#else

template <class T>
bool accumulateFusedIfAvailable(const Eigen::Ref<const MatrixT<T>>&, const Eigen::Ref<const VectorT<T>>&,
                                Eigen::Ref<MatrixT<T>>, Eigen::Ref<VectorT<T>>) {
  return false;
}

#endif

}  // namespace

template <class T>
void accumulateNormalEquations(const Eigen::Ref<const MatrixT<T>>& jacobian,
                               const Eigen::Ref<const VectorT<T>>& residuals, Eigen::Ref<MatrixT<T>> hessian,
                               Eigen::Ref<VectorT<T>> gradient) {
  assert(residuals.size() == jacobian.rows());
  assert(hessian.rows() == jacobian.cols() && hessian.cols() == jacobian.cols());
  assert(gradient.size() == jacobian.cols());

  if (!accumulateFusedIfAvailable<T>(jacobian, residuals, hessian, gradient)) {
    accumulateByDotProducts<T>(jacobian, residuals, hessian, gradient);
  }
}

template void accumulateNormalEquations<double>(const Eigen::Ref<const MatrixT<double>>&,
                                                const Eigen::Ref<const VectorT<double>>&, Eigen::Ref<MatrixT<double>>,
                                                Eigen::Ref<VectorT<double>>);
template void accumulateNormalEquations<float>(const Eigen::Ref<const MatrixT<float>>&,
                                               const Eigen::Ref<const VectorT<float>>&, Eigen::Ref<MatrixT<float>>,
                                               Eigen::Ref<VectorT<float>>);

}  // namespace moptim
