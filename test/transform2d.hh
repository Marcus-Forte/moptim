#pragma once

#include <gtest/gtest.h>

#include "IModel.hh"
#include "IOptimizer.hh"

using namespace moptim;

/**
 * @brief 2D Point distance model
 *
 */
struct Point2Distance : public IJacobianModel {
  void setup(const double* x) final {
    transform_.setIdentity();
    transform_.rotate(x[2]);
    transform_.translate(Eigen::Vector2d{x[0], x[1]});
  }

  void f(const double* input, const double* measurement, double* f_x) final {
    Eigen::Map<const Eigen::Vector2d> target{measurement};
    Eigen::Map<const Eigen::Vector2d> source{input};
    Eigen::Map<Eigen::Vector2d> transformed_point{f_x};

    transformed_point = target - transform_ * source;
  }

  void df(const double* input, const double* measurement, double* df_x) final {
    throw std::runtime_error("Unimplemented 2d point jacobian!");
    // const auto* target = reinterpret_cast<const Eigen::Vector2d*>(measurement);
    // const auto* source = reinterpret_cast<const Eigen::Vector2d*>(input);
    // auto* jacobian = reinterpret_cast<Eigen::Matrix<double, 2, 3>*>(df_x);

    // *jacobian = jacobian(*source, *target);

    // Eigen::Matrix<double, 2, 3> jac;
    // const auto cos_theta = std::cos(x_[2]);
    // const auto sin_theta = std::sin(x_[2]);
    // jac(0, 0) = -1;
    // jac(0, 1) = 0;
    // jac(0, 2) = -cos_theta * source[0] + sin_theta * source[1];

    // jac(1, 0) = 0;
    // jac(1, 1) = -1;
    // jac(1, 2) = sin_theta * source[0] + cos_theta * source[1];
  }

  Eigen::Affine2d transform_;
};

class TestTransform2D : public ::testing::Test {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_;
  std::shared_ptr<IOptimizer<double>> solver_;
};