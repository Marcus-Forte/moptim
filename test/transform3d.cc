#include "transform3d.hh"

#include <gtest/gtest.h>

#include <random>

static std::vector<Eigen::Vector3d> generateCloud3D(size_t n_points) {
  double x;
  double y;
  double z;
  std::vector<Eigen::Vector3d> pointcloud;
  std::string line;
  std::uniform_real_distribution<double> dist(-100, 100);
  std::mt19937 engine;

  pointcloud.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    x = dist(engine);
    y = dist(engine);
    z = dist(engine);
    pointcloud.emplace_back(x, y, z);
  }
  return pointcloud;
}

void TestTransform3D::SetUp() {
  pointcloud_ = generateCloud3D(GetParam());

  Eigen::AngleAxisd rollAngle(x0_ref[3], Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(x0_ref[4], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(x0_ref[5], Eigen::Vector3d::UnitZ());

  Eigen::Affine3d transform = Eigen::Affine3d::Identity();

  transform.translate(Eigen::Vector3d{x0_ref[0], x0_ref[1], x0_ref[2]});
  transform.rotate(rollAngle * pitchAngle * yawAngle);
  std::transform(pointcloud_.begin(), pointcloud_.end(), std::back_inserter(transformed_pointcloud_),
                 [&](const Eigen::Vector3d& pt) { return transform * pt; });
}
