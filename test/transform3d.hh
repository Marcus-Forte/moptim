#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "IModel.hh"
#include "IOptmizer.hh"

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
    Eigen::Map<Eigen::Vector3d> input_map{const_cast<double*>(input)};
    Eigen::Map<Eigen::Vector3d> measurement_map{const_cast<double*>(measurement)};
    Eigen::Map<Eigen::Vector3d> f_x_map{f_x};
    f_x_map = measurement_map - transform_ * input_map;
  }

  void df(const double* input, const double* measurement, double* df_x) final {
    Eigen::Map<Eigen::Vector3d> input_map{const_cast<double*>(input)};
    Eigen::Map<Eigen::Vector3d> measurement_map{const_cast<double*>(measurement)};
    // f_x_map = measurement_map - transform_ * input_map;
    df_x[0] = 222;
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