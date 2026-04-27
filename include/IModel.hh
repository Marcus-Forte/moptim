#pragma once

#include <mdspan>
#include <span>

template <class T>
class IModel {
 public:
  virtual ~IModel() = default;

  /**
   * @brief Compute residuals for all elements given parameters x.
   *
   * @param x               Parameters [param_dim]
   * @param inputs          All inputs  [num_elements × input_dim]
   * @param observations    All observations [num_elements × observation_dim]
   * @param residuals_out   Residuals to fill [num_elements × observation_dim]
   */
  virtual void residuals(std::span<const T> x, std::mdspan<const T, std::dextents<size_t, 2>> inputs,
                         std::mdspan<const T, std::dextents<size_t, 2>> observations,
                         std::mdspan<T, std::dextents<size_t, 2>> residuals_out) = 0;
};

template <class T>
class IJacobianModel : public IModel<T> {
 public:
  ~IJacobianModel() override = default;

  /**
   * @brief Compute Jacobians for all elements given parameters x.
   *        Output layout: row i holds the flattened [observation_dim × param_dim] Jacobian
   *        for element i (row-major), i.e. col j = dR_ij/dx_k stored as j*param_dim+k.
   *
   * @param x               Parameters [param_dim]
   * @param inputs          All inputs  [num_elements × input_dim]
   * @param observations    All observations [num_elements × observation_dim]
   * @param jacobians_out   Jacobians to fill [num_elements × (observation_dim * param_dim)]
   */
  virtual void jacobians(std::span<const T> x, std::mdspan<const T, std::dextents<size_t, 2>> inputs,
                         std::mdspan<const T, std::dextents<size_t, 2>> observations,
                         std::mdspan<T, std::dextents<size_t, 2>> jacobians_out) = 0;
};

// ─── Convenience bases — the loop lives here, not in user code ────────────────
//
// Inherit from ElementModel<Derived, T> or ElementJacobianModel<Derived, T>
// and implement only the single-element math.  The library owns the batch loop
// and is free to parallelise, vectorise, or dispatch to a GPU without any
// change to user code.
//
// CRTP is used so the per-element call is a direct non-virtual static dispatch
// — the compiler can inline it and auto-vectorise the loop across elements.
//
// Usage:
//   struct MyModel : ElementJacobianModel<MyModel, double> {
//     void residual(span<const double> x, span<const double> in,
//                   span<const double> obs, span<double> res) { ... }
//     void jacobian(span<const double> x, span<const double> in,
//                   span<const double> obs, span<double> jac) { ... }
//   };

/**
 * @brief Convenience base for numerical-Jacobian models (CRTP).
 *
 * Implement residual() with the per-element formula.
 * The batch residuals() loop is provided and marked final.
 */
template <class Derived, class T>
class ElementModel : public IModel<T> {
 public:
  void residuals(std::span<const T> x, std::mdspan<const T, std::dextents<size_t, 2>> inputs,
                 std::mdspan<const T, std::dextents<size_t, 2>> observations,
                 std::mdspan<T, std::dextents<size_t, 2>> residuals_out) final {
    const size_t n = inputs.extent(0);
    const size_t in_dim = inputs.extent(1);
    const size_t obs_dim = observations.extent(1);
    const size_t res_dim = residuals_out.extent(1);
    for (size_t i = 0; i < n; ++i) {
      static_cast<Derived*>(this)->residual(x, {inputs.data_handle() + i * in_dim, in_dim},
                                            {observations.data_handle() + i * obs_dim, obs_dim},
                                            {residuals_out.data_handle() + i * res_dim, res_dim});
    }
  }
};

/**
 * @brief Convenience base for analytical-Jacobian models (CRTP).
 *
 * Implement residual() and jacobian() with per-element formulas.
 * Both batch loops are provided and marked final.
 */
template <class Derived, class T>
class ElementJacobianModel : public IJacobianModel<T> {
 public:
  void residuals(std::span<const T> x, std::mdspan<const T, std::dextents<size_t, 2>> inputs,
                 std::mdspan<const T, std::dextents<size_t, 2>> observations,
                 std::mdspan<T, std::dextents<size_t, 2>> residuals_out) final {
    const size_t n = inputs.extent(0);
    const size_t in_dim = inputs.extent(1);
    const size_t obs_dim = observations.extent(1);
    const size_t res_dim = residuals_out.extent(1);
    for (size_t i = 0; i < n; ++i) {
      static_cast<Derived*>(this)->residual(x, {inputs.data_handle() + i * in_dim, in_dim},
                                            {observations.data_handle() + i * obs_dim, obs_dim},
                                            {residuals_out.data_handle() + i * res_dim, res_dim});
    }
  }

  void jacobians(std::span<const T> x, std::mdspan<const T, std::dextents<size_t, 2>> inputs,
                 std::mdspan<const T, std::dextents<size_t, 2>> observations,
                 std::mdspan<T, std::dextents<size_t, 2>> jacobians_out) final {
    const size_t n = inputs.extent(0);
    const size_t in_dim = inputs.extent(1);
    const size_t obs_dim = observations.extent(1);
    const size_t jac_dim = jacobians_out.extent(1);
    for (size_t i = 0; i < n; ++i) {
      static_cast<Derived*>(this)->jacobian(x, {inputs.data_handle() + i * in_dim, in_dim},
                                            {observations.data_handle() + i * obs_dim, obs_dim},
                                            {jacobians_out.data_handle() + i * jac_dim, jac_dim});
    }
  }
};