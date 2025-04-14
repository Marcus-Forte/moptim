#pragma once

#include <gtest/gtest.h>

#include "IOptmizer.hh"

/**
 * @brief 2D Point distance model
 *
 */
struct Point2Distance {
  Point2Distance(const Eigen::VectorXd& x) : x_(x) {
    transform_.setIdentity();
    transform_.rotate(x_[2]);
    transform_.translate(Eigen::Vector2d{x_[0], x_[1]});
  }

  Eigen::Vector2d operator()(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    return target - transform_ * source;
  }

  Eigen::Matrix<double, 2, 3> jacobian(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    Eigen::Matrix<double, 2, 3> jac;
    const auto cos_theta = std::cos(x_[2]);
    const auto sin_theta = std::sin(x_[2]);
    jac(0, 0) = -1;
    jac(0, 1) = 0;
    jac(0, 2) = -cos_theta * source[0] + sin_theta * source[1];

    jac(1, 0) = 0;
    jac(1, 1) = -1;
    jac(1, 2) = sin_theta * source[0] + cos_theta * source[1];
    return jac;
  }

  Eigen::Affine2d transform_;
  Eigen::Vector3d x_;
};

class Transform2D : public ::testing::Test {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_;
  std::shared_ptr<IOptimizer> solver_;
};