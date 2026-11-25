#pragma once

#include <cmath>
#include <limits>

namespace moptim {

/// The built-in robust losses, plus `Custom` for any other LossFunction subclass.
enum class LossKind { Trivial, Huber, Cauchy, GemanMcClure, Custom };

/**
 * @brief Plain-data description of a built-in loss, so that the cost's workers can evaluate it
 * without a virtual call (see detail::evaluateLoss()).
 *
 * | kind         | p0         | p1        |
 * |--------------|------------|-----------|
 * | Trivial      | unused     | unused    |
 * | Huber        | delta      | delta^2   |
 * | Cauchy       | scale^2    | unused    |
 * | GemanMcClure | kappa      | unused    |
 * | Custom       | -          | -         |
 */
template <class T>
struct LossDescriptor {
  LossKind kind{LossKind::Custom};
  T p0{};
  T p1{};
};

namespace detail {

template <class T>
T lossSqrt(T v) {
  return std::sqrt(v);
}
template <class T>
T lossLog(T v) {
  return std::log(v);
}

/// The formula of one built-in loss, selected at compile time. The built-ins' evaluate() call it
/// directly (no switch on a descriptor in the hot loops of the costs); evaluateLoss() dispatches to it.
template <LossKind Kind, class T>
inline void evaluateKnownLoss(T p0, T p1, T squared_norm, T out[3]) {
  if constexpr (Kind == LossKind::Trivial) {
    out[0] = squared_norm;
    out[1] = T{1};
    out[2] = T{0};
  } else if constexpr (Kind == LossKind::Huber) {
    const T delta = p0;
    const T delta_sq = p1;
    if (squared_norm <= delta_sq) {
      out[0] = squared_norm;
      out[1] = T{1};
      out[2] = T{0};
    } else {
      const T sqrt_s = lossSqrt(squared_norm);
      out[0] = T{2} * delta * sqrt_s - delta_sq;
      out[1] = delta / sqrt_s;
      out[2] = -delta / (T{2} * squared_norm * sqrt_s);
    }
  } else if constexpr (Kind == LossKind::Cauchy) {
    const T scale_sq = p0;
    const T sum = scale_sq + squared_norm;
    const T inv_sum = T{1} / sum;
    out[0] = scale_sq * lossLog(sum / scale_sq);
    out[1] = scale_sq * inv_sum;
    out[2] = -scale_sq * inv_sum * inv_sum;
  } else {
    static_assert(Kind == LossKind::GemanMcClure);
    const T kappa = p0;
    const T sum = kappa + squared_norm;
    const T inv_sum = T{1} / sum;
    out[0] = T{0.5} * squared_norm * inv_sum;
    out[1] = T{0.5} * kappa * inv_sum * inv_sum;
    out[2] = -kappa * inv_sum * inv_sum * inv_sum;
  }
}

/**
 * Evaluates the loss `descriptor` describes at `squared_norm`: `out[0] = rho(s)`, `out[1] = rho'(s)`,
 * `out[2] = rho''(s)`. No virtuals, exceptions or allocation; the built-in
 * losses' `evaluate()` run the same formulas (evaluateKnownLoss), so both share the arithmetic.
 *
 * A `Custom` descriptor has no formula: all three outputs are NaN.
 */
template <class T>
void evaluateLoss(const LossDescriptor<T>& descriptor, T squared_norm, T out[3]) {
  switch (descriptor.kind) {
    case LossKind::Trivial:
      evaluateKnownLoss<LossKind::Trivial>(descriptor.p0, descriptor.p1, squared_norm, out);
      return;
    case LossKind::Huber:
      evaluateKnownLoss<LossKind::Huber>(descriptor.p0, descriptor.p1, squared_norm, out);
      return;
    case LossKind::Cauchy:
      evaluateKnownLoss<LossKind::Cauchy>(descriptor.p0, descriptor.p1, squared_norm, out);
      return;
    case LossKind::GemanMcClure:
      evaluateKnownLoss<LossKind::GemanMcClure>(descriptor.p0, descriptor.p1, squared_norm, out);
      return;
    case LossKind::Custom:
      break;
  }
  out[0] = out[1] = out[2] = std::numeric_limits<T>::quiet_NaN();
}

}  // namespace detail

/**
 * @brief Robust loss (M-estimator) interface.
 *
 * Given the squared norm `s = ||r||^2` of a residual block, `evaluate` fills:
 *   out[0] = rho(s)
 *   out[1] = rho'(s)
 *   out[2] = rho''(s)
 *
 * Costs apply the first-order (IRLS) correction: the residual block and its
 * jacobian rows are scaled by sqrt(rho'(s)), and rho(s) is accumulated as the
 * cost contribution of that element instead of the raw s. This is the same
 * scheme used by e.g. KISS-ICP's Geman-McClure weighting. rho''(s) is
 * reported for completeness / potential second-order corrections but is not
 * used by the CPU costs at this time.
 */
template <class T>
class LossFunction {
 public:
  virtual ~LossFunction() = default;
  virtual void evaluate(T squared_norm, T out[3]) const = 0;

  /// Plain-data form of this loss, read by the costs at every evaluation. `Custom` (the default) is not
  /// evaluated by the costs.
  virtual LossDescriptor<T> descriptor() const { return {}; }
};

/// Identity loss: rho(s) = s. Equivalent to not setting a loss function.
template <class T>
class TrivialLoss : public LossFunction<T> {
 public:
  void evaluate(T squared_norm, T out[3]) const override {
    detail::evaluateKnownLoss<LossKind::Trivial>(T{}, T{}, squared_norm, out);
  }
  LossDescriptor<T> descriptor() const override { return describe(); }

 private:
  static LossDescriptor<T> describe() { return {LossKind::Trivial, T{}, T{}}; }
};

/**
 * @brief Huber loss.
 *
 * rho(s) = s                          for s <= delta^2
 * rho(s) = 2*delta*sqrt(s) - delta^2  for s >  delta^2
 */
template <class T>
class HuberLoss : public LossFunction<T> {
 public:
  explicit HuberLoss(T delta) : delta_(delta), delta_sq_(delta * delta) {}

  void setScale(T delta) {
    delta_ = delta;
    delta_sq_ = delta * delta;
  }
  T scale() const { return delta_; }

  void evaluate(T squared_norm, T out[3]) const override {
    detail::evaluateKnownLoss<LossKind::Huber>(delta_, delta_sq_, squared_norm, out);
  }
  LossDescriptor<T> descriptor() const override { return describe(); }

 private:
  LossDescriptor<T> describe() const { return {LossKind::Huber, delta_, delta_sq_}; }

  T delta_;
  T delta_sq_;
};

/**
 * @brief Cauchy (Lorentzian) loss.
 *
 * rho(s) = scale^2 * log(1 + s/scale^2)
 */
template <class T>
class CauchyLoss : public LossFunction<T> {
 public:
  explicit CauchyLoss(T scale) : scale_sq_(scale * scale) {}

  void setScale(T scale) { scale_sq_ = scale * scale; }
  T scale() const { return std::sqrt(scale_sq_); }

  void evaluate(T squared_norm, T out[3]) const override {
    detail::evaluateKnownLoss<LossKind::Cauchy>(scale_sq_, T{}, squared_norm, out);
  }
  LossDescriptor<T> descriptor() const override { return describe(); }

 private:
  LossDescriptor<T> describe() const { return {LossKind::Cauchy, scale_sq_, T{}}; }

  T scale_sq_;
};

/**
 * @brief Geman-McClure loss, as used e.g. by KISS-ICP.
 *
 * rho(s) = (s / 2) / (kappa + s)
 */
template <class T>
class GemanMcClureLoss : public LossFunction<T> {
 public:
  explicit GemanMcClureLoss(T kappa) : kappa_(kappa) {}

  void setScale(T kappa) { kappa_ = kappa; }
  T scale() const { return kappa_; }

  void evaluate(T squared_norm, T out[3]) const override {
    detail::evaluateKnownLoss<LossKind::GemanMcClure>(kappa_, T{}, squared_norm, out);
  }
  LossDescriptor<T> descriptor() const override { return describe(); }

 private:
  LossDescriptor<T> describe() const { return {LossKind::GemanMcClure, kappa_, T{}}; }

  T kappa_;
};

}  // namespace moptim
