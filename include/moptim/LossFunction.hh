#pragma once

#include <cmath>

namespace moptim {

/**
 * @brief Robust kernel (M-estimator) interface.
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
};

/// Identity loss: rho(s) = s. Equivalent to not setting a loss function.
template <class T>
class TrivialLoss : public LossFunction<T> {
 public:
  void evaluate(T squared_norm, T out[3]) const override {
    out[0] = squared_norm;
    out[1] = T{1};
    out[2] = T{0};
  }
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
    if (squared_norm <= delta_sq_) {
      out[0] = squared_norm;
      out[1] = T{1};
      out[2] = T{0};
    } else {
      const T sqrt_s = std::sqrt(squared_norm);
      out[0] = T{2} * delta_ * sqrt_s - delta_sq_;
      out[1] = delta_ / sqrt_s;
      out[2] = -delta_ / (T{2} * squared_norm * sqrt_s);
    }
  }

 private:
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
    const T sum = scale_sq_ + squared_norm;
    const T inv_sum = T{1} / sum;
    out[0] = scale_sq_ * std::log(sum / scale_sq_);
    out[1] = scale_sq_ * inv_sum;
    out[2] = -scale_sq_ * inv_sum * inv_sum;
  }

 private:
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
    const T sum = kappa_ + squared_norm;
    const T inv_sum = T{1} / sum;
    out[0] = T{0.5} * squared_norm * inv_sum;
    out[1] = T{0.5} * kappa_ * inv_sum * inv_sum;
    out[2] = -kappa_ * inv_sum * inv_sum * inv_sum;
  }

 private:
  T kappa_;
};

}  // namespace moptim
