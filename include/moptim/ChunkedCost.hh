#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "moptim/ICost.hh"
#include "moptim/IModel.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NormalEquations.hh"

namespace moptim {

/**
 * @brief CPU cost that streams its elements through cache-sized chunks.
 *
 * The residuals and jacobian rows of a chunk of elements are evaluated into small buffers that stay
 * in the L1 data cache and are folded into J^T J, J^T b and the cost right away. The full jacobian
 * (num_elements * observation_dim rows) is never stored, so memory use and construction time do not
 * grow with the number of elements.
 *
 * A derived class only says how the jacobian rows of a chunk are obtained (evaluateJacobian()).
 * This class owns everything else: the element loop, the optional noise model (scalar weights or
 * information matrices), the optional robust kernel and the accumulation.
 */
template <class Model, class T>
  requires ResidualModel<Model, T>
class ChunkedCost : public ICost<T> {
 public:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  T computeCost(const T* x) final {
    model_.setState(x);

    T cost{0};
    for (size_t first = 0; first < num_elements_; first += chunk_elements_) {
      const size_t count = std::min(chunk_elements_, num_elements_ - first);
      auto residuals = residuals_.head(count * observation_dim_);

      evaluateResiduals(model_, x, first, count, residuals.data());
      // A jacobian without columns: only the cost of the chunk is wanted.
      cost += isRobustified() ? robustify(first, count, residuals, jacobian_.block(0, 0, residuals.size(), 0))
                              : residuals.squaredNorm();
    }
    return cost;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) final {
    model_.setState(x);
    prepareJacobian(x);

    Eigen::Map<MatrixT> hessian(JTJ, tangent_dim_, tangent_dim_);
    Eigen::Map<VectorT> gradient(JTb, tangent_dim_);
    hessian.setZero();
    gradient.setZero();
    // Only the leading active_param_dim_ block is populated; the trailing
    // rows/columns (tangent dimensions this cost does not depend on) stay zero.
    auto active_hessian = hessian.topLeftCorner(active_param_dim_, active_param_dim_);
    auto active_gradient = gradient.head(active_param_dim_);

    cost = T{0};
    for (size_t first = 0; first < num_elements_; first += chunk_elements_) {
      const size_t count = std::min(chunk_elements_, num_elements_ - first);
      auto residuals = residuals_.head(count * observation_dim_);
      auto jacobian = jacobian_.topRows(count * observation_dim_);

      evaluateResiduals(model_, x, first, count, residuals.data());
      evaluateJacobian(x, first, count, residuals, jacobian);
      cost += isRobustified() ? robustify(first, count, residuals, jacobian) : residuals.squaredNorm();
      accumulateNormalEquations<T>(jacobian, residuals, active_hessian, active_gradient);
    }

    // The accumulation fills the lower triangle only.
    active_hessian.template triangularView<Eigen::StrictlyUpper>() = active_hessian.transpose();
  }

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
   * `weights.size()` must equal the number of elements; an empty vector removes the weights.
   */
  void setWeights(std::vector<T> weights) {
    assert(weights.empty() || weights.size() == num_elements_);
    weights_ = std::move(weights);
    information_factors_.resize(0, 0);
  }

  /**
   * @brief Set a full per-element information matrix (inverse covariance),
   * e.g. propagated from a Kalman filter's measurement/process noise. Each
   * matrix is `observation_dim x observation_dim` and is factored once
   * (Cholesky) at set-time; residuals/jacobians are whitened with L^T.
   * Clears any previously set scalar weights.
   * `information.size()` must equal the number of elements; an empty vector removes the matrices.
   */
  void setInformation(const std::vector<MatrixT>& information) {
    assert(information.empty() || information.size() == num_elements_);
    weights_.clear();
    information_factors_.resize(observation_dim_ * observation_dim_, information.size());
    for (size_t e = 0; e < information.size(); ++e) {
      assert(static_cast<size_t>(information[e].rows()) == observation_dim_ &&
             static_cast<size_t>(information[e].cols()) == observation_dim_);
      Eigen::Map<MatrixT>(information_factors_.col(e).data(), observation_dim_, observation_dim_) =
          information[e].llt().matrixL();
    }
  }

 protected:
  ChunkedCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
              typename ICost<T>::ParamSpace space, Model model, size_t active_param_dim)
      : ICost<T>(input_dim, observation_dim, std::move(space), num_elements, active_param_dim),
        model_(std::move(model)),
        input_(input),
        observations_(observations),
        chunk_elements_(chunkElements(num_elements_, observation_dim_, active_param_dim_)),
        residuals_(chunk_elements_ * observation_dim_),
        scales_(chunk_elements_ * observation_dim_),
        jacobian_(chunk_elements_ * observation_dim_, active_param_dim_) {}

  /// Called once per computeLinearSystem(), after the model's state was set to `x` and before the
  /// first chunk.
  virtual void prepareJacobian(const T* /*x*/) {}

  /**
   * @brief Writes the jacobian rows of elements [first, first + count).
   *
   * @param residuals Residuals of those elements at `x`, `count * observation_dim` entries.
   * @param jacobian `count * observation_dim` rows by `active_param_dim` columns. Row
   *   `e * observation_dim + i` receives the derivatives of residual `i` of element `first + e`
   *   with respect to the step (the parameters themselves when they are Euclidean).
   */
  virtual void evaluateJacobian(const T* x, size_t first, size_t count, const Eigen::Ref<const VectorT>& residuals,
                                Eigen::Ref<MatrixT> jacobian) = 0;

  /// Evaluates `model` on elements [first, first + count), writing `count * observation_dim`
  /// residuals. `x` is the parameter vector `model` had its state set to.
  ///
  /// `residuals` must be a buffer of this cost, which nothing else points into. Declaring it so
  /// lets the compiler keep the model's state in registers across the loop instead of reloading
  /// it after every residual is stored. The function is kept out of line because GCC only honours
  /// that promise for the parameters of the function it is compiling, not once the loop has been
  /// inlined into its caller.
  [[gnu::noinline]] void evaluateResiduals(Model& model, const T* x, size_t first, size_t count,
                                           T* __restrict__ residuals) const {
    const T* input = input_ + first * input_dim_;
    const T* observation = observations_ + first * observation_dim_;
    for (size_t e = 0; e < count; ++e) {
      model.residual(x, input, observation, residuals);
      input += input_dim_;
      observation += observation_dim_;
      residuals += observation_dim_;
    }
  }

  const T* input(size_t element) const { return input_ + element * input_dim_; }
  const T* observation(size_t element) const { return observations_ + element * observation_dim_; }

  /// Largest number of residual rows evaluateJacobian() is asked for at once.
  size_t chunkRows() const { return chunk_elements_ * observation_dim_; }

  /// True if all the elements fit in one chunk, so evaluateJacobian() is called at most once per
  /// linear system.
  bool isSingleChunk() const { return num_elements_ <= chunk_elements_; }

  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::tangent_dim_;
  using ICost<T>::num_elements_;
  using ICost<T>::active_param_dim_;

  Model model_;

 private:
  // Budget for the residual and jacobian block of one chunk. It leaves room in a 32 KiB L1 data
  // cache, the smallest in common use, for the chunk's inputs and observations.
  static constexpr size_t kChunkBytes = 16 * 1024;

  static size_t chunkElements(size_t num_elements, size_t observation_dim, size_t active_param_dim) {
    const size_t bytes_per_element = observation_dim * (active_param_dim + 1) * sizeof(T);
    const size_t fitting = kChunkBytes / std::max<size_t>(bytes_per_element, 1);
    return std::clamp<size_t>(fitting, 1, std::max<size_t>(num_elements, 1));
  }

  /// True if any noise model or robust kernel is configured. Checked once per chunk, so the default
  /// (unconfigured) case never touches the per-element robust path.
  bool isRobustified() const { return loss_ || !weights_.empty() || information_factors_.cols() > 0; }

  /**
   * @brief Applies the noise model and the robust kernel to a chunk in place and returns its cost.
   *
   * With W_e the whitening of element e (sqrt of its weight, or L_e^T for an information matrix
   * L_e L_e^T) and s_e = ||W_e r_e||^2, the element costs rho(s_e), and its residual block and
   * jacobian rows become sqrt(rho'(s_e)) * W_e * (r_e, J_e): the first-order (IRLS) correction.
   * Without a kernel rho(s) = s.
   *
   * `jacobian` may have no columns, in which case only the cost is computed and the residuals are
   * left whitened but unscaled.
   */
  T robustify(size_t first, size_t count, Eigen::Ref<VectorT> residuals, Eigen::Ref<MatrixT> jacobian) {
    const Eigen::Index dim = observation_dim_;
    const bool has_information = information_factors_.cols() > 0;
    const bool has_weights = !weights_.empty();

    T cost{0};
    for (size_t e = 0; e < count; ++e) {
      T* residual = residuals.data() + e * dim;
      if (has_information) {
        const T* factor = information_factors_.col(first + e).data();
        whiten(factor, dim, residual);
        for (Eigen::Index col = 0; col < jacobian.cols(); ++col) {
          whiten(factor, dim, jacobian.col(col).data() + e * dim);
        }
      }

      T squared_norm{0};
      for (Eigen::Index i = 0; i < dim; ++i) {
        squared_norm += residual[i] * residual[i];
      }

      T scale{1};
      if (has_weights) {
        const T weight = weights_[first + e];
        squared_norm *= weight;
        scale = std::sqrt(weight);
      }
      if (loss_) {
        T rho[3];
        loss_->evaluate(squared_norm, rho);
        cost += rho[0];
        scale *= std::sqrt(rho[1]);
      } else {
        cost += squared_norm;
      }
      std::fill_n(scales_.data() + e * dim, dim, scale);
    }

    // The scales of all the rows of the chunk are applied at once, one vectorized pass per column.
    if (jacobian.cols() > 0 && (has_weights || loss_)) {
      const auto scales = scales_.head(residuals.size()).array();
      residuals.array() *= scales;
      for (Eigen::Index col = 0; col < jacobian.cols(); ++col) {
        jacobian.col(col).array() *= scales;
      }
    }
    return cost;
  }

  /// column <- L^T * column, in place, for the `dim` entries of one element. `factor` is the
  /// column-major lower factor L. Row i of L^T only reads entries >= i, so going down the rows
  /// never reads an entry that was already overwritten.
  static void whiten(const T* factor, Eigen::Index dim, T* column) {
    for (Eigen::Index row = 0; row < dim; ++row) {
      T sum{0};
      for (Eigen::Index k = row; k < dim; ++k) {
        sum += factor[k + row * dim] * column[k];
      }
      column[row] = sum;
    }
  }

  const T* input_;
  const T* observations_;

  const size_t chunk_elements_;
  VectorT residuals_;  ///< Residuals of the current chunk.
  VectorT scales_;     ///< Per-row scale of the current chunk, used by robustify().
  MatrixT jacobian_;   ///< Jacobian rows of the current chunk, one column per active parameter.

  std::shared_ptr<LossFunction<T>> loss_;
  std::vector<T> weights_;
  /// Column e holds the lower Cholesky factor of element e's information matrix, in column-major
  /// order. No columns when no information matrices are set.
  MatrixT information_factors_;
};

}  // namespace moptim
