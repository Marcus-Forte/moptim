#pragma once

// Manifolds and models used by the manifold tests and benchmarks. The library ships no geometry:
// these are a consumer's conventions, written out here.
//
//   CircleManifold<T>  S1 stored as [cos, sin] (ambient 2, tangent 1), plus(x, d) rotates by d.
//   SE3Manifold        a pose stored as the 12 numbers of the 3x4 matrix [R | t] in column-major order
//                      (ambient 12), tangent [rho, omega] (6), left increment
//                      plus(T, delta) = Exp(delta) * T.

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>

#include "moptim/Manifold.hh"

namespace manifold_models {

// ─── Circle ───────────────────────────────────────────────────────────────────────────────────

template <class T>
class CircleManifold : public moptim::IManifold<T> {
 public:
  size_t ambientDim() const override { return 2; }
  size_t tangentDim() const override { return 1; }

  void plus(const T* x, const T* delta, T* x_plus) const override {
    const T c = std::cos(delta[0]);
    const T s = std::sin(delta[0]);
    x_plus[0] = c * x[0] - s * x[1];
    x_plus[1] = s * x[0] + c * x[1];
  }
};

/// Rotation angle between 2D point pairs: input = p (2), observation = q (2),
/// residual = q - R(x) p with R built from x = [cos, sin].
template <class T>
struct CircleRotationModel {
  void setState(const T* x) {
    c_ = x[0];
    s_ = x[1];
  }

  void residual(const T* /*x*/, const T* input, const T* obs, T* res) const {
    res[0] = obs[0] - (c_ * input[0] - s_ * input[1]);
    res[1] = obs[1] - (s_ * input[0] + c_ * input[1]);
  }

  /// d residual / d (angle step) = -perp(R p), one tangent dimension: jac[i] for residual i.
  void jacobian(const T* /*x*/, const T* input, const T* /*obs*/, T* jac) const {
    const T rx = c_ * input[0] - s_ * input[1];
    const T ry = s_ * input[0] + c_ * input[1];
    jac[0] = ry;
    jac[1] = -rx;
  }

  T c_ = 1;
  T s_ = 0;
};

// ─── SE(3) ────────────────────────────────────────────────────────────────────────────────────

inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return m;
}

/// Exp of se(3) with delta = [rho, omega]: rotation R = Exp(omega), translation V(omega) rho.
inline void se3Exp(const double* delta, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
  const Eigen::Vector3d rho(delta[0], delta[1], delta[2]);
  const Eigen::Vector3d omega(delta[3], delta[4], delta[5]);
  const double theta = omega.norm();
  const Eigen::Matrix3d W = hat(omega);
  double a, b, c;  // R = I + a W + b W^2, V = I + b W + c W^2
  if (theta < 1e-5) {
    a = 1 - theta * theta / 6;
    b = 0.5 - theta * theta / 24;
    c = 1.0 / 6 - theta * theta / 120;
  } else {
    a = std::sin(theta) / theta;
    b = (1 - std::cos(theta)) / (theta * theta);
    c = (1 - a) / (theta * theta);
  }
  R = Eigen::Matrix3d::Identity() + a * W + b * W * W;
  t = (Eigen::Matrix3d::Identity() + b * W + c * W * W) * rho;
}

class SE3Manifold : public moptim::IManifold<double> {
 public:
  size_t ambientDim() const override { return 12; }
  size_t tangentDim() const override { return 6; }

  void plus(const double* x, const double* delta, double* x_plus) const override {
    Eigen::Matrix3d dR;
    Eigen::Vector3d dt;
    se3Exp(delta, dR, dt);
    const Eigen::Map<const Eigen::Matrix3d> R(x);
    const Eigen::Map<const Eigen::Vector3d> t(x + 9);
    Eigen::Map<Eigen::Matrix3d>{x_plus} = dR * R;
    Eigen::Map<Eigen::Vector3d>{x_plus + 9} = dR * t + dt;
  }
};

inline Eigen::Matrix<double, 12, 1> packPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Eigen::Matrix<double, 12, 1> x;
  Eigen::Map<Eigen::Matrix3d>{x.data()} = R;
  x.tail<3>() = t;
  return x;
}

/// The pose Exp(delta) * identity, as 12 numbers.
inline Eigen::Matrix<double, 12, 1> poseFromTangent(const Eigen::Matrix<double, 6, 1>& delta) {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  se3Exp(delta.data(), R, t);
  return packPose(R, t);
}

/// Common state of the SE(3) models: the pose of the current evaluation point.
struct SE3State {
  void setState(const double* x) {
    R_ = Eigen::Map<const Eigen::Matrix3d>(x);
    t_ = Eigen::Map<const Eigen::Vector3d>(x + 9);
  }
  Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_ = Eigen::Vector3d::Zero();
};

/// Point-to-plane: input = [p, n] (6), observation = d (1), residual = d - n . (R p + t).
/// jacobian() is the derivative with respect to a LEFT increment: [-n, -(q x n)], q = R p + t.
struct PointToPlane : SE3State {
  void residual(const double* /*x*/, const double* input, const double* obs, double* res) const {
    const Eigen::Map<const Eigen::Vector3d> p(input), n(input + 3);
    res[0] = obs[0] - n.dot(R_ * p + t_);
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    const Eigen::Map<const Eigen::Vector3d> p(input), n(input + 3);
    const Eigen::Vector3d q = R_ * p + t_;
    Eigen::Map<Eigen::Matrix<double, 6, 1>> g(jac);
    g.head<3>() = -n;
    g.tail<3>() = -q.cross(n);
  }
};

/// The same residual with the jacobian of a RIGHT increment, T Exp(delta): [-R^T n, -(p x R^T n)].
/// Wrong for SE3Manifold, whose plus is a left increment.
struct PointToPlaneRightIncrement : PointToPlane {
  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    const Eigen::Map<const Eigen::Vector3d> p(input), n(input + 3);
    const Eigen::Vector3d m = R_.transpose() * n;
    Eigen::Map<Eigen::Matrix<double, 6, 1>> g(jac);
    g.head<3>() = -m;
    g.tail<3>() = -p.cross(m);
  }
};

/// Point-to-point: input = p (3), observation = target (3), residual = target - (R p + t).
/// jacobian() is the 6 x 3 transposed jacobian of a left increment: [-I, hat(q)].
struct PointToPoint : SE3State {
  void residual(const double* /*x*/, const double* input, const double* obs, double* res) const {
    Eigen::Map<Eigen::Vector3d>{res} =
        Eigen::Map<const Eigen::Vector3d>(obs) - (R_ * Eigen::Map<const Eigen::Vector3d>(input) + t_);
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    const Eigen::Vector3d q = R_ * Eigen::Map<const Eigen::Vector3d>(input) + t_;
    Eigen::Matrix<double, 3, 6> J;
    J.leftCols<3>() = -Eigen::Matrix3d::Identity();
    J.rightCols<3>() = hat(q);
    Eigen::Map<Eigen::Matrix<double, 6, 3>>{jac} = J.transpose();
  }
};

}  // namespace manifold_models
