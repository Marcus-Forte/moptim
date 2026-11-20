#pragma once
// The Point3Distance model and the point cloud of test/bench_moptim.cc, copied unchanged so the
// spike measures the same problem.

// Eigen's NEON backend includes <arm_neon.h>, which crashes the DPC++ native_cpu device front end
// ("unexpected Neon vector element type" while mangling vaddq_u64). The device pass of a translation
// unit therefore gets scalar Eigen, while the host pass keeps whatever the build asked for (NEON by
// default). The model's data layout does not depend on this (see the static_assert below).
#if defined(__SYCL_DEVICE_ONLY__) && !defined(EIGEN_DONT_VECTORIZE) && !defined(EIGEN_VECTORIZE_GENERIC)
#define EIGEN_DONT_VECTORIZE
#define EIGEN_ARM64_USE_SVE
#endif
#include <Eigen/Dense>
#include <cstddef>
#include <random>
#include <vector>

namespace spike {

// 3D rigid alignment, residual = target - R(roll, pitch, yaw) * (source + t).
// input_dim = observation_dim = 3, param_dim = 6.
struct Point3Distance {
  Eigen::Affine3d transform_{Eigen::Affine3d::Identity()};
  Eigen::Matrix3d Rx_{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d Ry_{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_{Eigen::Vector3d::Zero()};

  void setState(const double* x) {
    t_ = Eigen::Vector3d{x[0], x[1], x[2]};
    const Eigen::AngleAxisd roll(x[3], Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch(x[4], Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw(x[5], Eigen::Vector3d::UnitZ());
    Rx_ = roll.toRotationMatrix();
    Ry_ = pitch.toRotationMatrix();
    R_ = Rx_ * Ry_ * yaw.toRotationMatrix();
    transform_.setIdentity();
    transform_.rotate(roll * pitch * yaw);
    transform_.translate(t_);
  }

  void residual(const double* /*x*/, const double* input, const double* obs, double* res) const {
    Eigen::Map<const Eigen::Vector3d> source{input};
    Eigen::Map<const Eigen::Vector3d> target{obs};
    Eigen::Map<Eigen::Vector3d> result{res};
    result = target - transform_ * source;
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    using Vector3d = Eigen::Vector3d;
    const Vector3d s = Eigen::Map<const Vector3d>(input) + t_;
    const Vector3d p = R_ * s;

    Eigen::Map<Eigen::Matrix<double, 6, 3>> jac_t(jac);
    jac_t.topRows<3>() = -R_.transpose();
    jac_t.row(3) = Vector3d(0.0, p[2], -p[1]).transpose();
    const Vector3d q = Rx_.transpose() * p;
    jac_t.row(4) = -(Rx_ * Vector3d(q[2], 0.0, -q[0])).transpose();
    const Vector3d r_vec = Ry_.transpose() * q;
    jac_t.row(5) = -(Rx_ * Ry_ * Vector3d(-r_vec[1], r_vec[0], 0.0)).transpose();
  }
};

// The host and the device pass may differ in Eigen's vectorization (see above), so the layout the host
// writes and the kernel reads must not depend on it.
static_assert(sizeof(Point3Distance) == 128 + 3 * 72 + 24, "layout of the model differs between passes");
// Variants that pin the alignment (EIGEN_MAX_STATIC_ALIGN_BYTES=16) assert it in both passes.
#ifdef SPIKE_EXPECTED_ALIGN
static_assert(alignof(Point3Distance) == SPIKE_EXPECTED_ALIGN, "alignment of the model differs between passes");
#endif

struct Cloud {
  std::vector<double> source;  // num_elements * 3, model input
  std::vector<double> target;  // num_elements * 3, observations
};

inline Cloud makeCloud(size_t num_elements) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
  std::normal_distribution<double> noise(0.0, 0.05);

  const Eigen::Affine3d pose =
      Eigen::Translation3d(0.1, -0.2, 0.3) * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ());

  Cloud cloud;
  cloud.source.resize(num_elements * 3);
  cloud.target.resize(num_elements * 3);
  for (size_t i = 0; i < num_elements; ++i) {
    const Eigen::Vector3d point(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d moved = pose * point + Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
    Eigen::Map<Eigen::Vector3d>{&cloud.source[i * 3]} = point;
    Eigen::Map<Eigen::Vector3d>{&cloud.target[i * 3]} = moved;
  }
  return cloud;
}

}  // namespace spike
