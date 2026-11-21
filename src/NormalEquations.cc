#include "moptim/NormalEquations.hh"

#include <cassert>

#include "moptim/detail/NormalEquationsKernel.hh"

namespace moptim {
namespace {

template <class T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T>
using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

}  // namespace

template <class T>
void accumulateNormalEquations(const Eigen::Ref<const MatrixT<T>>& jacobian,
                               const Eigen::Ref<const VectorT<T>>& residuals, Eigen::Ref<MatrixT<T>> hessian,
                               Eigen::Ref<VectorT<T>> gradient) {
  assert(residuals.size() == jacobian.rows());
  assert(hessian.rows() == jacobian.cols() && hessian.cols() == jacobian.cols());
  assert(gradient.size() == jacobian.cols());

  detail::accumulateNormalEquationsRaw<T>(jacobian.data(), jacobian.outerStride(), residuals.data(), jacobian.rows(),
                                          jacobian.cols(), hessian.data(), hessian.outerStride(), gradient.data());
}

template void accumulateNormalEquations<double>(const Eigen::Ref<const MatrixT<double>>&,
                                                const Eigen::Ref<const VectorT<double>>&, Eigen::Ref<MatrixT<double>>,
                                                Eigen::Ref<VectorT<double>>);
template void accumulateNormalEquations<float>(const Eigen::Ref<const MatrixT<float>>&,
                                               const Eigen::Ref<const VectorT<float>>&, Eigen::Ref<MatrixT<float>>,
                                               Eigen::Ref<VectorT<float>>);

}  // namespace moptim
