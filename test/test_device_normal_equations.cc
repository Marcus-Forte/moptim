// The normal-equation kernel header (include/moptim/detail/NormalEquationsKernel.hh) run by a SYCL
// kernel against the same function called on the host, and against a dense Eigen reference.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "moptim/ExecutionContext.hh"
#include "moptim/detail/NormalEquationsKernel.hh"
#include "moptim/detail/Usm.hh"

using namespace moptim;

namespace {

template <class T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T>
using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
bool bitwiseEqual(const T* a, const T* b, size_t n) {
  return std::memcmp(a, b, n * sizeof(T)) == 0;
}

template <class T>
void checkShape(ExecutionContext& context, std::mt19937& rng, Eigen::Index rows, Eigen::Index cols) {
  std::uniform_real_distribution<T> entry(T(-1), T(1));
  // A jacobian with a column stride larger than its row count, as a chunk of a cost has.
  const Eigen::Index stride = rows + 3;
  const size_t h_size = static_cast<size_t>(cols * cols);

  auto jacobian = detail::makeSharedArray<T>(context.queue(), static_cast<size_t>(stride * cols));
  auto residuals = detail::makeSharedArray<T>(context.queue(), static_cast<size_t>(rows));
  auto hessian = detail::makeSharedArray<T>(context.queue(), h_size);
  auto gradient = detail::makeSharedArray<T>(context.queue(), static_cast<size_t>(cols));
  for (Eigen::Index i = 0; i < stride * cols; ++i) jacobian.get()[i] = entry(rng);
  for (Eigen::Index i = 0; i < rows; ++i) residuals.get()[i] = entry(rng);

  // Both start from the same non-zero H and g: the kernel adds into them, lower triangle only.
  std::vector<T> h_start(h_size), g_start(static_cast<size_t>(cols));
  for (T& v : h_start) v = entry(rng);
  for (T& v : g_start) v = entry(rng);
  std::vector<T> h_host = h_start, g_host = g_start;

  const T* j = jacobian.get();
  const T* r = residuals.get();
  detail::accumulateNormalEquationsRaw<T>(j, stride, r, rows, cols, h_host.data(), cols, g_host.data());

  std::copy(h_start.begin(), h_start.end(), hessian.get());
  std::copy(g_start.begin(), g_start.end(), gradient.get());
  T* h_out = hessian.get();
  T* g_out = gradient.get();
  detail::launch(context, 1, [=](sycl::nd_item<1>) {
    detail::accumulateNormalEquationsRaw<T>(j, stride, r, rows, cols, h_out, cols, g_out);
  });

  ASSERT_TRUE(bitwiseEqual(h_host.data(), h_out, h_size)) << "H, rows " << rows << " cols " << cols;
  ASSERT_TRUE(bitwiseEqual(g_host.data(), g_out, static_cast<size_t>(cols))) << "g, rows " << rows << " cols " << cols;

  // Dense reference.
  Eigen::Map<const MatrixT<T>, 0, Eigen::OuterStride<>> jm(j, rows, cols, Eigen::OuterStride<>(stride));
  Eigen::Map<const VectorT<T>> rv(r, rows);
  const MatrixT<T> start = Eigen::Map<const MatrixT<T>>(h_start.data(), cols, cols);
  const MatrixT<T> expected_h = start + MatrixT<T>(jm.transpose() * jm);
  const VectorT<T> expected_g = Eigen::Map<const VectorT<T>>(g_start.data(), cols) + VectorT<T>(jm.transpose() * rv);
  const T tol = sizeof(T) == 8 ? T(1e-12) : T(1e-4);
  const Eigen::Map<const MatrixT<T>> actual_h(h_out, cols, cols);
  for (Eigen::Index col = 0; col < cols; ++col) {
    for (Eigen::Index row = 0; row < cols; ++row) {
      if (row >= col) {
        ASSERT_NEAR(actual_h(row, col), expected_h(row, col), tol * (T(1) + std::abs(expected_h(row, col))))
            << "H(" << row << "," << col << ") rows " << rows << " cols " << cols;
      } else {
        ASSERT_EQ(actual_h(row, col), start(row, col)) << "strict upper triangle was written";
      }
    }
    ASSERT_NEAR(g_out[col], expected_g(col), tol * (T(1) + std::abs(expected_g(col)))) << "g(" << col << ")";
  }
}

template <class T>
void sweep() {
  ExecutionContext context(sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()});
  std::mt19937 rng(11);
  for (Eigen::Index cols = 1; cols <= 24; ++cols) {
    for (Eigen::Index rows = 1; rows <= 300; ++rows) {
      checkShape<T>(context, rng, rows, cols);
      if (::testing::Test::HasFatalFailure()) return;
    }
  }
}

}  // namespace

TEST(TestDeviceNormalEquations, DoubleKernelMatchesHost) { sweep<double>(); }
TEST(TestDeviceNormalEquations, FloatKernelMatchesHost) { sweep<float>(); }

TEST(TestDeviceNormalEquations, SquaredNormMatchesAcrossPasses) {
  ExecutionContext context(sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()});
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> entry(-1.0, 1.0);
  auto values = detail::makeSharedArray<double>(context.queue(), 100);
  auto out = detail::makeSharedArray<double>(context.queue(), 100);
  for (int i = 0; i < 100; ++i) values.get()[i] = entry(rng);
  const double* v = values.get();
  double* o = out.get();
  detail::launch(context, 1, [=](sycl::nd_item<1>) {
    for (size_t n = 0; n < 100; ++n) o[n] = detail::squaredNorm(v, n);
  });
  for (size_t n = 0; n < 100; ++n) {
    double reference = 0.0;
    for (size_t i = 0; i < n; ++i) reference += v[i] * v[i];
    EXPECT_EQ(o[n], detail::squaredNorm(v, n)) << n;
    EXPECT_NEAR(o[n], reference, 1e-13 * (1.0 + reference)) << n;
  }
}
