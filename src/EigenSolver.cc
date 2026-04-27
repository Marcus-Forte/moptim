#include "EigenSolver.hh"

#include <Eigen/Dense>
#include <cassert>

template <typename T>
void EigenSolver<T>::solve(std::span<const T> A, std::span<const T> b, std::span<T> x) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  const size_t dimensions = this->dimensions_;
  assert(A.size() == dimensions * dimensions);
  assert(b.size() == dimensions);
  assert(x.size() == dimensions);

  Eigen::Map<const MatrixT> matA(A.data(), dimensions, dimensions);
  Eigen::Map<const VectorT> vecb(b.data(), dimensions);
  Eigen::Map<VectorT> vecx(x.data(), dimensions);

  Eigen::LDLT<MatrixT> solver(matA);

  vecx = solver.solve(-vecb);
}

template class EigenSolver<double>;
template class EigenSolver<float>;