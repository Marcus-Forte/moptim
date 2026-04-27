#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "IModel.hh"
#include "IOptimizer.hh"

using namespace moptim;

/**
 * @brief 3D Point distance model
 *
 */
struct Point3Distance : public ElementJacobianModel<Point3Distance, double> {
  void residual(std::span<const double> x, std::span<const double> input, std::span<const double> obs,
                std::span<double> res) {
    Eigen::Affine3d transform;
    transform.setIdentity();
    Eigen::AngleAxisd rollAngle(x[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(x[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(x[5], Eigen::Vector3d::UnitZ());
    transform.rotate(rollAngle * pitchAngle * yawAngle);
    transform.translate(Eigen::Vector3d{x[0], x[1], x[2]});
    Eigen::Map<const Eigen::Vector3d> source{input.data()};
    Eigen::Map<const Eigen::Vector3d> target{obs.data()};
    Eigen::Map<Eigen::Vector3d> result{res.data()};
    result = target - transform * source;
  }

  void jacobian(std::span<const double> /*x*/, std::span<const double> /*input*/, std::span<const double> /*obs*/,
                std::span<double> /*jac*/) {
    throw std::runtime_error("Unimplemented 3d point jacobian!");
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