#include "EigenSolver.hh"

#include <Eigen/Dense>

template <typename T>
void EigenSolver<T>::solve(const T* A, const T* b, T* x) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  const size_t dimensions = this->dimensions_;
  Eigen::Map<const MatrixT> matA(A, dimensions, dimensions);
  Eigen::Map<const VectorT> vecb(b, dimensions);
  Eigen::Map<VectorT> vecx(x, dimensions);

  Eigen::LDLT<MatrixT> solver(matA);

  vecx = solver.solve(-vecb);
}

template class EigenSolver<double>;
template class EigenSolver<float>;