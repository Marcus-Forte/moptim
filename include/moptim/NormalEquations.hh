#pragma once

#include <Eigen/Core>

namespace moptim {

/**
 * @brief Folds a block of jacobian rows into the normal equations: `H += J^T J` and `g += J^T r`.
 *
 * Only the lower triangle of `H` is written. The strict upper triangle is left untouched, so a
 * caller that needs the full symmetric matrix mirrors it once, after the last block.
 *
 * Defined in `src/NormalEquations.cc` and instantiated there for `float` and `double`.
 *
 * @param jacobian J, one row per residual and one column per parameter.
 * @param residuals r, `jacobian.rows()` entries.
 * @param hessian H, `jacobian.cols()` rows and columns.
 * @param gradient g, `jacobian.cols()` entries.
 */
template <class T>
void accumulateNormalEquations(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& jacobian,
                               const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& residuals,
                               Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> hessian,
                               Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> gradient);

}  // namespace moptim
