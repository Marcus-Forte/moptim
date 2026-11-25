#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>

#include "moptim/NormalEquations.hh"

using namespace moptim;

namespace {

template <class T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T>
using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
MatrixT<T> randomMatrix(Eigen::Index rows, Eigen::Index cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> entry(T(-1), T(1));
  MatrixT<T> matrix(rows, cols);
  for (Eigen::Index i = 0; i < matrix.size(); ++i) matrix(i) = entry(rng);
  return matrix;
}

/// Relative tolerance for sums of a few hundred products of entries in [-1, 1].
template <class T>
T tolerance() {
  return sizeof(T) == sizeof(double) ? T(1e-12) : T(1e-4);
}

template <class T>
void expectNear(const MatrixT<T>& actual, const MatrixT<T>& expected) {
  EXPECT_LE((actual - expected).norm(), tolerance<T>() * (T(1) + expected.norm())) << "actual\n"
                                                                                   << actual << "\nexpected\n"
                                                                                   << expected;
}

template <class T>
MatrixT<T> lowerTriangle(const MatrixT<T>& matrix) {
  return matrix.template triangularView<Eigen::Lower>();
}

}  // namespace

template <class T>
class NormalEquationsTest : public ::testing::Test {};

using Scalars = ::testing::Types<double, float>;
TYPED_TEST_SUITE(NormalEquationsTest, Scalars);

// Every column count goes through its own routine (fused, in one or more passes, up to a dozen
// columns on AArch64; dot products beyond), and the row counts cover the empty block, blocks
// shorter than a SIMD vector and blocks with a remainder.
TYPED_TEST(NormalEquationsTest, MatchesDenseProducts) {
  using T = TypeParam;

  for (Eigen::Index cols = 1; cols <= 16; ++cols) {
    for (const Eigen::Index rows : {0, 1, 2, 3, 5, 8, 17, 64, 291}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " cols=" << cols);
      const MatrixT<T> jacobian = randomMatrix<T>(rows, cols, 1);
      const VectorT<T> residuals = randomMatrix<T>(rows, 1, 2);

      MatrixT<T> hessian = MatrixT<T>::Zero(cols, cols);
      VectorT<T> gradient = VectorT<T>::Zero(cols);
      accumulateNormalEquations<T>(jacobian, residuals, hessian, gradient);

      const MatrixT<T> expected_hessian = jacobian.transpose() * jacobian;
      const VectorT<T> expected_gradient = jacobian.transpose() * residuals;
      // Only the lower triangle is written.
      expectNear<T>(hessian, lowerTriangle<T>(expected_hessian));
      expectNear<T>(gradient, expected_gradient);
    }
  }
}

TYPED_TEST(NormalEquationsTest, AccumulatesOnTopOfExistingContent) {
  using T = TypeParam;
  const Eigen::Index rows = 50;

  for (const Eigen::Index cols : {3, 6, 9, 14}) {
    SCOPED_TRACE(testing::Message() << "cols=" << cols);
    const MatrixT<T> jacobian = randomMatrix<T>(rows, cols, 3);
    const VectorT<T> residuals = randomMatrix<T>(rows, 1, 4);

    // The two halves of the rows, added one after the other, give the whole.
    MatrixT<T> hessian = MatrixT<T>::Zero(cols, cols);
    VectorT<T> gradient = VectorT<T>::Zero(cols);
    accumulateNormalEquations<T>(jacobian.topRows(20), residuals.head(20), hessian, gradient);
    accumulateNormalEquations<T>(jacobian.bottomRows(30), residuals.tail(30), hessian, gradient);

    const MatrixT<T> expected_hessian = jacobian.transpose() * jacobian;
    const VectorT<T> expected_gradient = jacobian.transpose() * residuals;
    expectNear<T>(hessian, lowerTriangle<T>(expected_hessian));
    expectNear<T>(gradient, expected_gradient);
  }
}

// The cost classes pass blocks of larger buffers: the jacobian is the top of a taller chunk buffer
// and the hessian is the leading block of the full parameter-sized matrix.
TYPED_TEST(NormalEquationsTest, WritesOnlyInsideTheBlocksItIsGiven) {
  using T = TypeParam;
  const Eigen::Index rows = 37;
  const T untouched = T(-7);

  for (const Eigen::Index cols : {4, 8, 15}) {
    SCOPED_TRACE(testing::Message() << "cols=" << cols);
    const MatrixT<T> chunk = randomMatrix<T>(rows + 11, cols, 5);
    const VectorT<T> chunk_residuals = randomMatrix<T>(rows + 11, 1, 6);

    MatrixT<T> full_hessian = MatrixT<T>::Constant(cols + 3, cols + 3, untouched);
    VectorT<T> full_gradient = VectorT<T>::Constant(cols + 3, untouched);
    full_hessian.topLeftCorner(cols, cols).setZero();
    full_gradient.head(cols).setZero();

    accumulateNormalEquations<T>(chunk.topRows(rows), chunk_residuals.head(rows),
                                 full_hessian.topLeftCorner(cols, cols), full_gradient.head(cols));

    const MatrixT<T> jacobian = chunk.topRows(rows);
    const VectorT<T> residuals = chunk_residuals.head(rows);
    const MatrixT<T> expected_hessian = jacobian.transpose() * jacobian;
    const VectorT<T> expected_gradient = jacobian.transpose() * residuals;
    expectNear<T>(full_hessian.topLeftCorner(cols, cols), lowerTriangle<T>(expected_hessian));
    expectNear<T>(full_gradient.head(cols), expected_gradient);

    EXPECT_TRUE((full_hessian.bottomRows(3).array() == untouched).all());
    EXPECT_TRUE((full_hessian.rightCols(3).array() == untouched).all());
    EXPECT_TRUE((full_gradient.tail(3).array() == untouched).all());
  }
}
