#pragma once

#include <concepts>

namespace moptim {

// ─── Concepts ─────────────────────────────────────────────────────────────────
//
// These concepts define the interfaces that Model types must satisfy.
// They are used as constraints on the template parameters of cost classes,
// giving clear compile-time errors when a model is missing a required method.
//
// ResidualModel<M, T>
//   Common to every cost. The model must provide:
//     void setState(const T* x);                                     — called once per cost evaluation
//     void residual(const T* x, const T* input, const T* obs, T* res);
//
// NumericalModel<M, T>
//   Required by NumericalCostForwardEuler and NumericalCostCentral.
//   A ResidualModel that can be copied. The cost keeps one copy of the model per finite-difference
//   evaluation point (1 + P_active forward, 1 + 2 P_active central) and calls setState() on each with
//   its own perturbed parameter vector, so copies must not share state that setState() or residual()
//   write to. Only `residual` is called on them, in a kernel (see below).
//
// Where a model runs
//   `residual` and `jacobian` run inside a launch of the ExecutionContext: inline on the calling thread
//   for a small cost, otherwise on several threads of a oneTBB arena at once. A model must therefore be
//   thread-safe in the sense below, and `residual`/`jacobian` must not throw (an exception of a worker
//   does reach the caller of the evaluation, but the partial sums of that launch are lost).
//   `setState` is the exception: it runs on the calling thread, once before every evaluation, and may do
//   anything (it is where an Eigen rotation or a transform is computed from `x`).
//
//   The model object is stored by the cost (constructed by moving the model passed to it) and destroyed
//   with the cost. It is never memcpy'd. Whatever it points to or references (a lookup table, an
//   external matrix) must stay valid, and be safe to read from several threads.
//
//   If `residual` and `jacobian` can be called on a `const` model they are called on the one model
//   from every work-item (thread), so they must not write to it (no `mutable` state). Otherwise each
//   work-item copy-constructs the model and uses its own copy, which the copy constructor and destructor
//   then run on that thread too (once per work-item per evaluation); a copy constructor must be safe to
//   run concurrently on the same source. A model that is neither const-callable nor copy-constructible
//   is evaluated by a single work-item, on one thread.
//
// AnalyticalModel<M, T>
//   Required by AnalyticalCost.
//   A ResidualModel that also provides the per-element jacobian:
//     void jacobian(const T* x, const T* input, const T* obs, T* jac);
//   `jac` is the transposed jacobian of the element, active_param_dim x observation_dim
//   in column-major order: jac[j + active_param_dim * i] = d residual[i] / d delta[j],
//   where delta is the step of the parameters, taken at delta = 0. For Euclidean parameters
//   x [+] delta = x + delta, so this is the ordinary derivative d residual[i] / d x[j]. On a
//   manifold it is the derivative with respect to the tangent vector in x [+] delta, in the
//   convention of the manifold's plus(), and j counts tangent dimensions, not entries of x.

template <class M, class T>
concept ResidualModel =
    std::move_constructible<M> && requires(M& m, const T* x, const T* input, const T* observation, T* residual) {
      { m.setState(x) };
      { m.residual(x, input, observation, residual) };
    };

/// `residual` can be called on a const model.
template <class M, class T>
concept ConstResidualModel = requires(const M& m, const T* x, const T* input, const T* observation, T* residual) {
  { m.residual(x, input, observation, residual) };
};

/// `jacobian` can be called on a const model.
template <class M, class T>
concept ConstJacobianModel = requires(const M& m, const T* x, const T* input, const T* observation, T* jacobian) {
  { m.jacobian(x, input, observation, jacobian) };
};

template <class M, class T>
concept NumericalModel = ResidualModel<M, T> && std::copy_constructible<M>;

template <class M, class T>
concept AnalyticalModel =
    ResidualModel<M, T> && requires(M& m, const T* x, const T* input, const T* observation, T* jacobian) {
      { m.jacobian(x, input, observation, jacobian) };
    };

}  // namespace moptim
