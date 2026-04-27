#pragma once

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "IModel.hh"
#include "IOptimizer.hh"

using namespace moptim;

/**
 * @brief 2D Point distance model
 *
 */
struct Point2Distance : public ElementJacobianModel<Point2Distance, double> {
  static void residual(std::span<const double> x, std::span<const double> input, std::span<const double> obs,
                std::span<double> res) {
    Eigen::Affine2d transform;
    transform.setIdentity();
    transform.rotate(x[2]);
    transform.translate(Eigen::Vector2d{x[0], x[1]});
    Eigen::Map<const Eigen::Vector2d> target{obs.data()};
    Eigen::Map<const Eigen::Vector2d> source{input.data()};
    Eigen::Map<Eigen::Vector2d> result{res.data()};
    result = target - transform * source;
  }

  static void jacobian(std::span<const double> x, std::span<const double> input, std::span<const double> /*obs*/,
                std::span<double> jac) {
    const auto c = std::cos(x[2]);
    const auto s = std::sin(x[2]);
    Eigen::Map<Eigen::Matrix<double, 3, 2>> jac_t(jac.data());
    const auto u = input[0] + x[0];
    const auto v = input[1] + x[1];
    jac_t(0, 0) = -c;
    jac_t(1, 0) = s;
    jac_t(2, 0) = s * u + c * v;
    jac_t(0, 1) = -s;
    jac_t(1, 1) = -c;
    jac_t(2, 1) = -c * u + s * v;
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