#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "moptim/ChunkedCost.hh"
#include "moptim/IModel.hh"

namespace moptim {

/// Finite-difference formula used by NumericalCost.
enum class DifferenceScheme {
  Forward,  ///< (r(x + h) - r(x)) / h: one extra evaluation per parameter, O(h) truncation error.
  Central,  ///< (r(x + h) - r(x - h)) / 2h: two per parameter, O(h^2) truncation error.
};

/**
 * @brief Cost whose jacobian is obtained by finite differences of the model's residuals.
 *
 * The residuals are evaluated at one or two extra points per active parameter: `x` with that
 * parameter stepped up, or up and down. A chunk of elements is run through all of these points
 * while its inputs and observations are still in cache, instead of making one pass over the whole
 * data set per parameter.
 *
 * When the elements span several chunks, every evaluation point gets its own copy of the model,
 * with its state set once per linear system. When they fit in a single chunk each point is visited
 * exactly once, so the cost's own model is moved from point to point and nothing is copied.
 *
 * On a manifold the evaluation points are `x [+] (+h e_p)` and `x [+] (-h e_p)` for each active
 * tangent dimension `p`, so the jacobian is the derivative with respect to the step at zero, the
 * same quantity an analytical model provides. The points are computed once per linear system and
 * then used exactly like the Euclidean ones. Without a manifold they are `x` with entry `p` stepped
 * in place.
 *
 * Use it through the NumericalCostForwardEuler and NumericalCostCentral aliases.
 */
template <class Model, class T, DifferenceScheme Scheme>
  requires NumericalModel<Model, T>
class NumericalCost : public ChunkedCost<Model, T> {
  using Base = ChunkedCost<Model, T>;

 public:
  using typename Base::MatrixT;
  using typename Base::VectorT;

  /// Euclidean parameters: `x` has `param_dim` entries.
  NumericalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                size_t param_dim, Model model = Model{}, size_t active_param_dim = 0)
      : NumericalCost(input, observations, num_elements, input_dim, observation_dim, Base::euclideanSpace(param_dim),
                      std::move(model), active_param_dim) {}

  /// Parameters on a manifold. `active_param_dim` counts leading tangent dimensions.
  NumericalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                std::shared_ptr<const IManifold<T>> manifold, Model model = Model{}, size_t active_param_dim = 0)
      : NumericalCost(input, observations, num_elements, input_dim, observation_dim, Base::spaceOf(std::move(manifold)),
                      std::move(model), active_param_dim) {}

 private:
  NumericalCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
                typename Base::ParamSpace space, Model model, size_t active_param_dim)
      : Base(input, observations, num_elements, input_dim, observation_dim, std::move(space), std::move(model),
             active_param_dim),
        point_models_(this->isSingleChunk() ? 0 : kPointsPerParam * active_param_dim_, this->model_),
        point_x_(this->manifold() ? 0 : param_dim_),
        point_table_(this->manifold() ? kPointsPerParam * active_param_dim_ * param_dim_ : 0),
        step_delta_(this->manifold() ? tangent_dim_ : 0),
        backward_residuals_(Scheme == DifferenceScheme::Central ? this->chunkRows() : 0) {}

  using Base::active_param_dim_;
  using Base::param_dim_;
  using Base::tangent_dim_;

  /// Evaluation points per active parameter. Point `kPointsPerParam * p` is `x + h e_p`; in the
  /// central scheme point `kPointsPerParam * p + 1` is `x - h e_p`.
  static constexpr size_t kPointsPerParam = Scheme == DifferenceScheme::Central ? 2 : 1;

  void prepareJacobian(const T* x) override {
    if (this->manifold()) {
      fillPointTable(x);
    } else {
      point_x_ = Eigen::Map<const VectorT>(x, param_dim_);
    }
    for (size_t point = 0; point < point_models_.size(); ++point) {
      if (this->manifold()) {
        point_models_[point].setState(point_table_.data() + point * param_dim_);
      } else {
        visitPoint(point, [&](const T* x_at_point) { point_models_[point].setState(x_at_point); });
      }
    }
  }

  /// Fills the jacobian one column per active parameter, from the residuals at its evaluation
  /// points.
  void evaluateJacobian(const T* /*x*/, size_t first, size_t count, const Eigen::Ref<const VectorT>& residuals,
                        Eigen::Ref<MatrixT> jacobian) override {
    for (size_t p = 0; p < active_param_dim_; ++p) {
      auto column = jacobian.col(p);
      evaluateAtPoint(kPointsPerParam * p, first, count, column.data());

      if constexpr (Scheme == DifferenceScheme::Central) {
        auto backward = backward_residuals_.head(column.size());
        evaluateAtPoint(kPointsPerParam * p + 1, first, count, backward.data());
        column = (column - backward) * inverse_span_;
      } else {
        column = (column - residuals) * inverse_span_;
      }
    }
  }

  /// Residuals of elements [first, first + count) at an evaluation point.
  void evaluateAtPoint(size_t point, size_t first, size_t count, T* residuals) {
    if (this->manifold()) {
      evaluateAtX(point, point_table_.data() + point * param_dim_, first, count, residuals);
    } else {
      visitPoint(point, [&](const T* x_at_point) { evaluateAtX(point, x_at_point, first, count, residuals); });
    }
  }

  /// Residuals of elements [first, first + count) at evaluation point `point`, whose parameter
  /// vector is `x_at_point`.
  void evaluateAtX(size_t point, const T* x_at_point, size_t first, size_t count, T* residuals) {
    if (point_models_.empty()) {
      // Single chunk: this is the only visit to the point, so the cost's own model goes there.
      // The next evaluation starts by setting its state again.
      this->model_.setState(x_at_point);
      this->evaluateResiduals(this->model_, x_at_point, first, count, residuals);
    } else {
      this->evaluateResiduals(point_models_[point], x_at_point, first, count, residuals);
    }
  }

  /// Calls `visit` with the parameter vector of an evaluation point, `x` stepped in place (no manifold).
  template <class Visit>
  void visitPoint(size_t point, Visit&& visit) {
    const size_t param = point / kPointsPerParam;
    const T step = point % kPointsPerParam == 0 ? step_ : -step_;

    const T original = point_x_[param];
    point_x_[param] = original + step;
    visit(point_x_.data());
    point_x_[param] = original;
  }

  /// Computes the evaluation points on the manifold: row `point` of the table is `x [+] delta`
  /// with `delta` the step of that point along its tangent dimension.
  void fillPointTable(const T* x) {
    step_delta_.setZero();
    for (size_t point = 0; point < kPointsPerParam * active_param_dim_; ++point) {
      const size_t param = point / kPointsPerParam;
      step_delta_[param] = point % kPointsPerParam == 0 ? step_ : -step_;
      this->manifold()->plus(x, step_delta_.data(), point_table_.data() + point * param_dim_);
      step_delta_[param] = T{0};
    }
  }

  const T step_ = std::sqrt(std::numeric_limits<T>::epsilon());
  /// 1 / distance between the two evaluation points of a parameter.
  const T inverse_span_ = T{1} / (static_cast<T>(kPointsPerParam) * step_);

  /// One model per evaluation point when the elements span several chunks, empty otherwise.
  std::vector<Model> point_models_;
  VectorT point_x_;             ///< Without a manifold: copy of `x`, stepped in place to each point in turn.
  VectorT point_table_;         ///< With a manifold: the evaluation points, one ambient-sized row each.
  VectorT step_delta_;          ///< With a manifold: scratch step of `tangent_dim` entries, zero but for one entry.
  VectorT backward_residuals_;  ///< Residuals at `x - h e_p` for the current chunk (central scheme).
};

}  // namespace moptim
