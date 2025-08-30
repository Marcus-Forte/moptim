#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "IModel.hh"
#include "IOptimizer.hh"

/**
 * @brief 3D Point distance model
 *
 */
struct Point3Distance : public IJacobianModel {
  void setup(const double* x) final {
    transform_.setIdentity();
    Eigen::AngleAxisd rollAngle(x[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(x[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(x[5], Eigen::Vector3d::UnitZ());
    transform_.rotate(rollAngle * pitchAngle * yawAngle);
    transform_.translate(Eigen::Vector3d{x[0], x[1], x[2]});
  }

  void f(const double* input, const double* measurement, double* f_x) final {
    Eigen::Map<const Eigen::Vector3d> input_map{input};
    Eigen::Map<const Eigen::Vector3d> measurement_map{measurement};
    Eigen::Map<Eigen::Vector3d> f_x_map{f_x};
    f_x_map = measurement_map - transform_ * input_map;
  }

  void df(const double* input, const double* measurement, double* df_x) final {
    Eigen::Map<const Eigen::Vector3d> input_map{input};
    Eigen::Map<const Eigen::Vector3d> measurement_map{measurement};
    // f_x_map = measurement_map - transform_ * input_map;
    throw std::runtime_error("Unimplemented 3d point jacobian!");
  }

  Eigen::Affine3d transform_;
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
  std::shared_ptr<IOptimizer> solver_;
};