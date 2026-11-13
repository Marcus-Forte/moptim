#include "moptim/EigenSolver.hh"

namespace moptim {

template <class T>
EigenSolver<T>::EigenSolver(size_t dimensions) : ISolver<T>(dimensions), factorization_(dimensions) {}

template <class T>
bool EigenSolver<T>::solve(const T* A, const T* b, T* x) {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  const size_t dimensions = this->dimensions_;

  Eigen::Map<const MatrixT> matA(A, dimensions, dimensions);
  Eigen::Map<const VectorT> vecb(b, dimensions);
  Eigen::Map<VectorT> vecx(x, dimensions);

  // LLT copies A into its own storage and factors the lower triangle in place.
  factorization_.compute(matA);
  if (factorization_.info() != Eigen::Success) {
    return false;
  }

  vecx = -vecb;
  factorization_.solveInPlace(vecx);
  // A NaN in A or b goes through the factorization unnoticed.
  return vecx.allFinite();
}

template class EigenSolver<double>;
template class EigenSolver<float>;

}  // namespace moptim
