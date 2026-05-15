#pragma once

#include <Eigen/Dense>

namespace moptim {

template <class T>
inline Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1>& v) {
  Eigen::Matrix<T, 3, 3> m;
  m << T{0}, -v.z(), v.y(), v.z(), T{0}, -v.x(), -v.y(), v.x(), T{0};
  return m;
}

template <class T>
inline Eigen::Matrix<T, 6, 1> se3Log(
    const Eigen::Transform<T, 3, Eigen::Affine>& T_ab) {
  const Eigen::Matrix<T, 3, 3> R = T_ab.rotation();
  const Eigen::Matrix<T, 3, 1> t = T_ab.translation();

  const Eigen::AngleAxis<T> aa(R);
  const T theta = aa.angle();
  const Eigen::Matrix<T, 3, 1> omega = aa.axis() * theta;

  Eigen::Matrix<T, 3, 3> V_inv;
  if (theta < T{1e-10}) {
    V_inv = Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    const Eigen::Matrix<T, 3, 3> omega_hat = hat(omega);
    V_inv = Eigen::Matrix<T, 3, 3>::Identity() - T{0.5} * omega_hat +
            (T{1} / (theta * theta)) *
                (T{1} - (theta * std::cos(theta / T{2})) /
                            (T{2} * std::sin(theta / T{2}))) *
                omega_hat * omega_hat;
  }

  Eigen::Matrix<T, 6, 1> xi;
  xi.template head<3>() = V_inv * t;
  xi.template tail<3>() = omega;
  return xi;
}

template <class Derived>
inline Eigen::Transform<typename Derived::Scalar, 3, Eigen::Affine> se3Exp(
    const Eigen::MatrixBase<Derived>& xi) {
  using T = typename Derived::Scalar;

  const Eigen::Matrix<T, 3, 1> rho = xi.template head<3>();
  const Eigen::Matrix<T, 3, 1> omega = xi.template tail<3>();
  const T theta = omega.norm();

  Eigen::Matrix<T, 3, 3> R;
  Eigen::Matrix<T, 3, 3> V;

  if (theta < T{1e-10}) {
    R = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    const Eigen::Matrix<T, 3, 3> omega_hat = hat(omega);
    const Eigen::Matrix<T, 3, 3> omega_hat2 = omega_hat * omega_hat;
    const T s = std::sin(theta) / theta;
    const T c = (T{1} - std::cos(theta)) / (theta * theta);

    R = Eigen::Matrix<T, 3, 3>::Identity() + s * omega_hat + c * omega_hat2;
    V = Eigen::Matrix<T, 3, 3>::Identity() + c * omega_hat +
        ((T{1} - s) / (theta * theta)) * omega_hat2;
  }

  Eigen::Transform<T, 3, Eigen::Affine> T_ab =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  T_ab.linear() = R;
  T_ab.translation() = V * rho;
  return T_ab;
}

}  // namespace moptim