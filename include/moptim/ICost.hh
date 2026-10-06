#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

#include "moptim/LossFunction.hh"

namespace moptim {

template <class T>
class ICost {
 public:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  ICost(const ICost&) = delete;
  virtual ~ICost() = default;
  /**
   * @param active_param_dim Number of leading parameters this cost depends on.
   *   The model still receives the full `x` (param_dim entries), but only the
   *   first `active_param_dim` are perturbed/differentiated and only the
   *   corresponding leading block of the linear system is populated. Passing
   *   `0` (the default) means the full `param_dim`.
   */
  ICost(size_t input_dim, size_t observation_dim, size_t param_dim, size_t num_elements, size_t active_param_dim = 0)
      : input_dim_(input_dim),
        observation_dim_(observation_dim),
        param_dim_(param_dim),
        num_elements_(num_elements),
        active_param_dim_(active_param_dim == 0 ? param_dim : active_param_dim) {
    assert(active_param_dim_ <= param_dim_);
  }

  /**
   * @brief Compute the cost given parameters x
   *
   * @param x Parameters
   * @return T Cost value
   */
  virtual T computeCost(const T* x) = 0;

  /**
   * @brief Compute the linear system: JTJ, JTb and cost
   *
   * @param x Parameters
   * @param JTJ Hessian (J^T * J)
   * @param JTb Gradient (J^T * b)
   * @param cost Cost value
   */
  virtual void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) = 0;

  /**
   * @brief Attach a robust kernel (M-estimator) applied to each element's
   * (optionally whitened) squared residual norm. Passing nullptr (default)
   * disables it, falling back to plain least-squares. Shared ownership
   * allows the same kernel to be reused/adapted (e.g. online scale updates,
   * a la KISS-ICP) across multiple costs.
   */
  void setLossFunction(std::shared_ptr<LossFunction<T>> loss) { loss_ = std::move(loss); }

  /**
   * @brief Set a scalar, isotropic weight per element (w_i = 1/sigma_i^2).
   * Equivalent to setInformation() with information_i = w_i * I, but cheaper.
   * Clears any previously set information matrices.
   * `weights.size()` must equal num_elements_.
   */
  void setWeights(std::vector<T> weights) {
    weights_ = std::move(weights);
    information_cholesky_.clear();
  }

  /**
   * @brief Set a full per-element information matrix (inverse covariance),
   * e.g. propagated from a Kalman filter's measurement/process noise. Each
   * matrix is `observation_dim_ x observation_dim_` and is factored once
   * (Cholesky) at set-time; residuals/jacobians are whitened with L^T.
   * Clears any previously set scalar weights.
   * `information.size()` must equal num_elements_.
   */
  void setInformation(const std::vector<MatrixT>& information) {
    weights_.clear();
    information_cholesky_.resize(information.size());
    for (size_t i = 0; i < information.size(); ++i) {
      information_cholesky_[i] = Eigen::LLT<MatrixT>(information[i]).matrixL();
    }
  }

 protected:
  bool hasInformation() const { return !information_cholesky_.empty(); }
  bool hasWeights() const { return !weights_.empty(); }
  bool hasLoss() const { return static_cast<bool>(loss_); }
  /// True if any noise model or robust kernel is configured. Costs check this
  /// once before their per-element loop so the default (unconfigured) case
  /// keeps running the original, branch-free loop body with no overhead.
  bool isRobustified() const { return hasInformation() || hasWeights() || hasLoss(); }
  const MatrixT& informationFactor(size_t element_index) const { return information_cholesky_[element_index]; }
  T weight(size_t element_index) const { return weights_[element_index]; }

  /**
   * @brief Evaluate the robust loss at a given (already whitened) squared
   * residual norm, returning the multiplicative scale to apply to both the
   * residual block and its jacobian rows, and writing the cost contribution
   * of this element to `cost_contribution`.
   */
  T applyLoss(T squared_norm, T& cost_contribution) const {
    if (!loss_) {
      cost_contribution = squared_norm;
      return T{1};
    }
    T rho[3];
    loss_->evaluate(squared_norm, rho);
    cost_contribution = rho[0];
    return std::sqrt(rho[1]);
  }

  /**
   * @brief Whiten (information/weights) and robust-loss-scale a single element's residual
   * block in-place, returning its cost contribution. Used by computeCost() implementations
   * that do not need the jacobian.
   */
  template <class ResidualBlock>
  T whitenAndScaleResidual(size_t element_index, ResidualBlock&& residual_block) const {
    if (hasInformation()) {
      residual_block = informationFactor(element_index).transpose() * residual_block;
    } else if (hasWeights()) {
      residual_block *= std::sqrt(weight(element_index));
    }
    T contribution;
    applyLoss(residual_block.squaredNorm(), contribution);
    return contribution;
  }

  /**
   * @brief Whiten (information/weights) and robust-loss-scale a single element's residual
   * and jacobian blocks in-place, returning its cost contribution. `jacobian_rows_are_residuals`
   * selects the jacobian layout: true when rows span the observation dimension (jacobian_block is
   * observation_dim x param_dim, as in the numerical costs), false when columns do (jacobian_block
   * is param_dim x observation_dim, i.e. J^T storage, as in AnalyticalCost).
   */
  template <class ResidualBlock, class JacobianBlock>
  T whitenAndScaleElement(size_t element_index, ResidualBlock&& residual_block, JacobianBlock&& jacobian_block,
                          bool jacobian_rows_are_residuals) const {
    if (hasInformation()) {
      const MatrixT& L = informationFactor(element_index);
      residual_block = L.transpose() * residual_block;
      if (jacobian_rows_are_residuals) {
        jacobian_block = (L.transpose() * jacobian_block).eval();
      } else {
        jacobian_block = (jacobian_block * L).eval();
      }
    } else if (hasWeights()) {
      const T w = std::sqrt(weight(element_index));
      residual_block *= w;
      jacobian_block *= w;
    }

    T contribution;
    const T scale = applyLoss(residual_block.squaredNorm(), contribution);
    residual_block *= scale;
    jacobian_block *= scale;
    return contribution;
  }

  const size_t input_dim_;
  const size_t observation_dim_;
  const size_t param_dim_;
  const size_t num_elements_;
  /// Number of leading parameters the cost actually depends on. Resolved from
  /// the constructor's active_param_dim (0 -> param_dim_), never exceeds param_dim_.
  const size_t active_param_dim_;

  std::shared_ptr<LossFunction<T>> loss_;
  std::vector<T> weights_;
  std::vector<MatrixT> information_cholesky_;
};
}  // namespace moptim
