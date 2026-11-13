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
//   A ResidualModel that can be copied. When the elements span more than one chunk,
//   the cost keeps one copy of the model per finite-difference evaluation point and
//   calls setState() on each with its own perturbed parameter vector, so copies must
//   not share state that setState() or residual() write to.
//
// AnalyticalModel<M, T>
//   Required by AnalyticalCost.
//   A ResidualModel that also provides the per-element jacobian:
//     void jacobian(const T* x, const T* input, const T* obs, T* jac);
//   `jac` is the transposed jacobian of the element, active_param_dim x observation_dim
//   in column-major order: jac[j + active_param_dim * i] = d residual[i] / d x[j].

template <class M, class T>
concept ResidualModel =
    std::move_constructible<M> && requires(M& m, const T* x, const T* input, const T* observation, T* residual) {
      { m.setState(x) };
      { m.residual(x, input, observation, residual) };
    };

template <class M, class T>
concept NumericalModel = ResidualModel<M, T> && std::copy_constructible<M>;

template <class M, class T>
concept AnalyticalModel =
    ResidualModel<M, T> && requires(M& m, const T* x, const T* input, const T* observation, T* jacobian) {
      { m.jacobian(x, input, observation, jacobian) };
    };

}  // namespace moptim
