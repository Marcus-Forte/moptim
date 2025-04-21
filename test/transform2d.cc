
#include "transform2d.hh"

#include <filesystem>
#include <fstream>
#include <random>

/**
 * @brief Reads ASC point cloud. Assume coordinates are the first two columns. Delimiter is whitespace.
 *
 * @param file
 * @return std::vector<Eigen::Vector2d>
 */
static std::vector<Eigen::Vector2d> read2DTxtScan(std::filesystem::path&& file) {
  if (!std::filesystem::exists(file)) {
    throw std::runtime_error("File does not exist: " + file.string());
  }
  std::ifstream pointcloud_file(file);

  double x;
  double y;
  double discard;
  std::vector<Eigen::Vector2d> pointcloud;
  std::string line;
  while (pointcloud_file >> x >> y >> discard >> discard >> discard >> discard >> discard >> discard >> discard) {
    pointcloud.emplace_back(x, y);
  }

  return pointcloud;
}

static void applyNoise(std::vector<Eigen::Vector2d>& pointcloud, double amplitude) {
  std::uniform_real_distribution<double> dist(-amplitude, amplitude);
  std::mt19937 engine;
  for (auto& point : pointcloud) {
    point[0] += dist(engine);
    point[1] += dist(engine);
  }
}

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