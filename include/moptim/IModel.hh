#pragma once

// ─── Concepts ─────────────────────────────────────────────────────────────────
//
// These concepts define the interfaces that Model types must satisfy.
// They are used as constraints on the template parameters of cost classes,
// giving clear compile-time errors when a model is missing a required method.
//
// NumericalModel<M, T>
//   Required by NumericalCostForwardEuler and NumericalCostCentral.
//   The model must provide:
//     void setState(const T* x);                                     — called once per cost evaluation
//     void residual(const T* x, const T* input, const T* obs, T* res);
//
// AnalyticalModel<M, T>
//   Required by AnalyticalCost.
//   The model must provide setState(), per-element residual() and jacobian().
//   Jacobian is filled in column-major order.

template <class M, class T>
concept NumericalModel = requires(M& m, const T* x, const T* input, const T* observation, T* residual) {
  { m.setState(x) };
  { m.residual(x, input, observation, residual) };
};

template <class M, class T>
concept AnalyticalModel = requires(M& m, const T* x, const T* input, const T* observation, T* residual, T* jacobian) {
  { m.setState(x) };
  { m.residual(x, input, observation, residual) };
  { m.jacobian(x, input, observation, jacobian) };
};
