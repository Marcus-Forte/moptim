#pragma once

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

/**
 * @brief 2D Point distance model
 *
 */
struct Point2Distance {
  Eigen::Affine2d transform_;
  double c_{1.0}, s_{0.0}, tx_{0.0}, ty_{0.0};

  void setState(const double* x) {
    tx_ = x[0];
    ty_ = x[1];
    c_ = std::cos(x[2]);
    s_ = std::sin(x[2]);
    transform_.setIdentity();
    transform_.rotate(x[2]);
    transform_.translate(Eigen::Vector2d{x[0], x[1]});
  }

  void residual(const double* /*x*/, const double* input, const double* obs, double* res) const {
    Eigen::Map<const Eigen::Vector2d> target{obs};
    Eigen::Map<const Eigen::Vector2d> source{input};
    Eigen::Map<Eigen::Vector2d> result{res};
    result = target - transform_ * source;
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    Eigen::Map<Eigen::Matrix<double, 3, 2>> jac_t(jac);
    const auto u = input[0] + tx_;
    const auto v = input[1] + ty_;
    jac_t(0, 0) = -c_;
    jac_t(1, 0) = s_;
    jac_t(2, 0) = s_ * u + c_ * v;
    jac_t(0, 1) = -s_;
    jac_t(1, 1) = -c_;
    jac_t(2, 1) = -c_ * u + s_ * v;
  }
};

class TestTransform2D : public ::testing::Test {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_;
};