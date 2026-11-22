#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include "moptim/Manifold.hh"

namespace moptim {

/**
 * @brief A least-squares term: the sum of squared residuals over `num_elements` elements, as a
 * function of a parameter vector of `param_dim` entries.
 *
 * The parameters either form a plain vector (Euclidean: `paramDim() == tangentDim()` and
 * `manifold()` is null) or live on a manifold, in which case `paramDim()` is its ambient size, the
 * number of entries of `x`, and `tangentDim()` the size of the linear system. A cost is always
 * built for the same space as the optimizer it is added to.
 */
template <class T>
class ICost {
 public:
  /**
   * Euclidean parameters.
   *
   * @param active_param_dim Number of leading parameters this cost depends on.
   *   The model still receives the full `x` (param_dim entries), but only the
   *   first `active_param_dim` are perturbed/differentiated and only the
   *   corresponding leading block of the linear system is populated. Passing
   *   `0` (the default) means the full `param_dim`.
   */
  ICost(size_t input_dim, size_t observation_dim, size_t param_dim, size_t num_elements, size_t active_param_dim = 0)
      : ICost(input_dim, observation_dim, ParamSpace{param_dim, param_dim, nullptr}, num_elements, active_param_dim) {}

  /**
   * Parameters on a manifold: `x` has `manifold->ambientDim()` entries, the linear system
   * `manifold->tangentDim()`.
   *
   * @param active_param_dim Number of leading *tangent* dimensions this cost depends on, at most
   *   `manifold->tangentDim()`; `0` (the default) means all of them. The model still receives the
   *   full `x`.
   */
  ICost(size_t input_dim, size_t observation_dim, std::shared_ptr<const IManifold<T>> manifold, size_t num_elements,
        size_t active_param_dim = 0)
      : ICost(input_dim, observation_dim, spaceOf(std::move(manifold)), num_elements, active_param_dim) {}

  ICost(const ICost&) = delete;
  ICost& operator=(const ICost&) = delete;
  virtual ~ICost() = default;

  /**
   * @brief Compute the cost given parameters x
   *
   * @param x Parameters (param_dim entries)
   * @return T Cost value
   */
  virtual T computeCost(const T* x) = 0;

  /**
   * @brief Compute the linear system: JTJ, JTb and cost
   *
   * Both outputs are overwritten, not accumulated into. JTJ is written in full (both triangles),
   * with zeros in the rows and columns of the tangent dimensions this cost does not depend on.
   * The jacobian is the derivative of the residuals with respect to the step `delta` at
   * `delta = 0` (`x [+] delta`), which for Euclidean parameters is the ordinary jacobian.
   *
   * @param x Parameters (param_dim entries)
   * @param JTJ Hessian (J^T * J), tangent_dim x tangent_dim
   * @param JTb Gradient (J^T * b), tangent_dim entries
   * @param cost Cost value, the same value computeCost(x) returns
   */
  virtual void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) = 0;

  /**
   * @brief Asynchronous form of computeLinearSystem(): starts the evaluation and returns, possibly
   * before it is done.
   *
   * `x` is read before the call returns, so the caller may reuse it. `JTJ`, `JTb` and `cost` are
   * written by the time wait() returns, not before, and must stay valid until then. The default
   * calls computeLinearSystem(), so a cost that does not override it behaves as before.
   *
   * At most one evaluation of a cost is outstanding: calling any of computeCost(),
   * computeLinearSystem(), submitCost() or submitLinearSystem() on a cost with one outstanding first
   * waits for it (its outputs are then written), then runs the new one.
   */
  virtual void submitLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) { computeLinearSystem(x, JTJ, JTb, *cost); }

  /// Asynchronous form of computeCost(); `cost` is written by the time wait() returns. Same rules as
  /// submitLinearSystem().
  virtual void submitCost(const T* x, T* cost) { *cost = computeCost(x); }

  /// Completes the outstanding submitLinearSystem() / submitCost(), if any. Does nothing when there
  /// is none, so it is safe to call more than once. Rethrows an error of the evaluation.
  virtual void wait() {}

  size_t inputDim() const { return input_dim_; }
  size_t observationDim() const { return observation_dim_; }
  /// Entries of `x` (the ambient dimension of the manifold, if any).
  size_t paramDim() const { return param_dim_; }
  /// Entries of a step: the size of JTJ and JTb. Equals paramDim() for Euclidean parameters.
  size_t tangentDim() const { return tangent_dim_; }
  /// Leading tangent dimensions the cost depends on.
  size_t activeParamDim() const { return active_param_dim_; }
  size_t numElements() const { return num_elements_; }
  /// The manifold the parameters live on, null for Euclidean parameters.
  const IManifold<T>* manifold() const { return manifold_.get(); }
  /// The same manifold, shared (e.g. to build another cost or a FusedCost on the same space).
  const std::shared_ptr<const IManifold<T>>& sharedManifold() const { return manifold_; }

 protected:
  /// Where the parameters live: `manifold` is null for Euclidean parameters.
  struct ParamSpace {
    size_t ambient_dim;
    size_t tangent_dim;
    std::shared_ptr<const IManifold<T>> manifold;
  };

  static ParamSpace euclideanSpace(size_t param_dim) { return {param_dim, param_dim, nullptr}; }

  static ParamSpace spaceOf(std::shared_ptr<const IManifold<T>> manifold) {
    assert(manifold);
    const size_t ambient = manifold->ambientDim();
    const size_t tangent = manifold->tangentDim();
    return {ambient, tangent, std::move(manifold)};
  }

  ICost(size_t input_dim, size_t observation_dim, ParamSpace space, size_t num_elements, size_t active_param_dim)
      : input_dim_(input_dim),
        observation_dim_(observation_dim),
        param_dim_(space.ambient_dim),
        tangent_dim_(space.tangent_dim),
        num_elements_(num_elements),
        active_param_dim_(active_param_dim == 0 ? space.tangent_dim : active_param_dim),
        manifold_(std::move(space.manifold)) {
    assert(active_param_dim_ <= tangent_dim_);
  }

  const size_t input_dim_;
  const size_t observation_dim_;
  const size_t param_dim_;
  const size_t tangent_dim_;
  /// Not const: a cost whose data can be replaced (ChunkedCost::setData()) changes it.
  size_t num_elements_;
  /// Number of leading tangent dimensions the cost actually depends on. Resolved from the
  /// constructor's active_param_dim (0 -> tangent_dim_), never exceeds tangent_dim_.
  const size_t active_param_dim_;
  const std::shared_ptr<const IManifold<T>> manifold_;
};
}  // namespace moptim
