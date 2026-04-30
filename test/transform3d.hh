#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "IOptimizer.hh"

using namespace moptim;

/**
 * @brief 3D Point distance model
 *
 */
struct Point3Distance {
  Eigen::Affine3d transform_;
  Eigen::Matrix3d Rx_, Ry_, R_;
  Eigen::Vector3d t_;

  void setState(const double* x) {
    t_ = Eigen::Vector3d{x[0], x[1], x[2]};
    const Eigen::AngleAxisd rollAngle(x[3], Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitchAngle(x[4], Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yawAngle(x[5], Eigen::Vector3d::UnitZ());
    Rx_ = rollAngle.toRotationMatrix();
    Ry_ = pitchAngle.toRotationMatrix();
    R_ = Rx_ * Ry_ * yawAngle.toRotationMatrix();
    transform_.setIdentity();
    transform_.rotate(rollAngle * pitchAngle * yawAngle);
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

    // p = R * (source + t), where t = [x[0], x[1], x[2]]
    const Vector3d s = Eigen::Map<const Vector3d>(input) + t_;
    const Vector3d p = R_ * s;

    // jac is [param_dim=6 x obs_dim=3] column-major: jac_t(j,i) = dr[i]/dx[j]
    Eigen::Map<Eigen::Matrix<double, 6, 3>> jac_t(jac);

    // Translation: dr[i]/dx[j] = -R[i,j]  →  top 3 rows = -R^T
    jac_t.topRows<3>() = -R_.transpose();

    // Roll (x[3]): dp/droll = [1,0,0] x p = [0, -p[2], p[1]]
    jac_t.row(3) = Vector3d(0.0, p[2], -p[1]).transpose();

    // Pitch (x[4]): dp/dpitch = Rx * ([0,1,0] x (Rx^T * p))
    const Vector3d q = Rx_.transpose() * p;
    jac_t.row(4) = -(Rx_ * Vector3d(q[2], 0.0, -q[0])).transpose();

    // Yaw (x[5]): dp/dyaw = Rx*Ry * ([0,0,1] x (Ry^T * Rx^T * p))
    const Vector3d r_vec = Ry_.transpose() * q;
    jac_t.row(5) = -(Rx_ * Ry_ * Vector3d(-r_vec[1], r_vec[0], 0.0)).transpose();
  }
};

/**
 * @brief Fixture for 3D transform tests.
 *
 */
class TestTransform3D : public ::testing::TestWithParam<int> {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.1, 0.1, 0, 0, 0}};
  std::vector<Eigen::Vector3d> transformed_pointcloud_;
  std::vector<Eigen::Vector3d> pointcloud_;
  std::shared_ptr<IOptimizer<double>> solver_;
};