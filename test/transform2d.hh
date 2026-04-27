#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Dense>

#include "IModel.hh"
#include "IOptimizer.hh"

using namespace moptim;

/**
 * @brief 2D Point distance model
 *
 */
struct Point2Distance : public IJacobianModel<double> {
  void setup(std::span<const double> x) final {
    x_ = Eigen::Map<const Eigen::Vector3d>(x.data());
    transform_.setIdentity();
    transform_.rotate(x_[2]);
    transform_.translate(Eigen::Vector2d{x_[0], x_[1]});
  }

  void f(std::span<const double> input, std::span<const double> measurement, std::span<double> f_x) final {
    Eigen::Map<const Eigen::Vector2d> target{measurement.data()};
    Eigen::Map<const Eigen::Vector2d> source{input.data()};
    Eigen::Map<Eigen::Vector2d> transformed_point{f_x.data()};

    transformed_point = target - transform_ * source;
  }

  void df(std::span<const double> input, std::span<const double> measurement, std::span<double> df_x) final {
    (void)measurement;
    Eigen::Map<Eigen::Matrix<double, 3, 2>> jac_t(df_x.data());

    const auto c = std::cos(x_[2]);
    const auto s = std::sin(x_[2]);
    const auto u = input[0] + x_[0];
    const auto v = input[1] + x_[1];

    jac_t(0, 0) = -c;
    jac_t(1, 0) = s;
    jac_t(2, 0) = s * u + c * v;

    jac_t(0, 1) = -s;
    jac_t(1, 1) = -c;
    jac_t(2, 1) = -c * u + s * v;
  }

  Eigen::Vector3d x_;
  Eigen::Affine2d transform_;
};

class TestTransform2D : public ::testing::Test {
 public:
  void SetUp() override;

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_;
};