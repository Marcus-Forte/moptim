#pragma once

#include <cassert>
#include <cstddef>

namespace moptim {

/**
 * @brief A least-squares term: the sum of squared residuals over `num_elements` elements, as a
 * function of a parameter vector of `param_dim` entries.
 */
template <class T>
class ICost {
 public:
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
   * with zeros in the rows and columns of the parameters this cost does not depend on.
   *
   * @param x Parameters (param_dim entries)
   * @param JTJ Hessian (J^T * J), param_dim x param_dim
   * @param JTb Gradient (J^T * b), param_dim entries
   * @param cost Cost value, the same value computeCost(x) returns
   */
  virtual void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) = 0;

  size_t inputDim() const { return input_dim_; }
  size_t observationDim() const { return observation_dim_; }
  size_t paramDim() const { return param_dim_; }
  size_t activeParamDim() const { return active_param_dim_; }
  size_t numElements() const { return num_elements_; }

 protected:
  const size_t input_dim_;
  const size_t observation_dim_;
  const size_t param_dim_;
  const size_t num_elements_;
  /// Number of leading parameters the cost actually depends on. Resolved from
  /// the constructor's active_param_dim (0 -> param_dim_), never exceeds param_dim_.
  const size_t active_param_dim_;
};
}  // namespace moptim
