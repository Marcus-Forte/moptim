#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "IOptmizer.hh"

/**
 * @brief 3D Point distance model
 *
 */
struct Point3Distance {
  Point3Distance(const Eigen::VectorXd& x) : x_(x) {
    transform_.setIdentity();
    Eigen::AngleAxisd rollAngle(x_[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(x_[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(x_[5], Eigen::Vector3d::UnitZ());
    transform_.rotate(rollAngle * pitchAngle * yawAngle);
    transform_.translate(Eigen::Vector3d{x_[0], x_[1], x_[2]});
  }

  Eigen::Vector3d operator()(const Eigen::Vector3d& source, const Eigen::Vector3d& target) const {
    return target - transform_ * source;
  }

  Eigen::Affine3d transform_;
  Eigen::Vector<double, 6> x_;
};

/**
 * @brief Fixture for 3D transform tests.
 *
 */
class Test3DTransform : public ::testing::TestWithParam<int> {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3, 0, 0, 0}};
  std::vector<Eigen::Vector3d> transformed_pointcloud_;
  std::vector<Eigen::Vector3d> pointcloud_;
  std::shared_ptr<IOptimizer> solver_;
};