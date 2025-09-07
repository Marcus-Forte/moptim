
#include "transform2d.hh"

#include "test_helper.hh"

/**
 * @brief Reads ASC point cloud. Assume coordinates are the first two columns. Delimiter is whitespace.
 *
 * @param file
 * @return std::vector<Eigen::Vector2d>
 */

void TestTransform2D::SetUp() {
  pointcloud_ = read2DTxtScan(TEST_PATH / std::filesystem::path("scan.txt"));

  Eigen::Rotation2D<double> rot(x0_ref[2]);
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();

  transform.translate(Eigen::Vector2d{x0_ref[0], x0_ref[1]});
  transform.rotate(rot);
  std::transform(pointcloud_.begin(), pointcloud_.end(), std::back_inserter(transformed_pointcloud_),
                 [&](const Eigen::Vector2d& pt) { return transform * pt; });
  ::applyNoise(transformed_pointcloud_, 0.01);
}