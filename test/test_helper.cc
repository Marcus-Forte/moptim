
#include "test_helper.hh"

#include <gtest/gtest.h>

#include <fstream>
#include <random>

void compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance) {
  ASSERT_EQ(m1.size(), m2.size());

  for (int i = 0; i < m1.size(); ++i) {
    EXPECT_NEAR(m1(i), m2(i), tolerance);
  }
}

std::vector<Eigen::Vector2d> read2DTxtScan(std::filesystem::path&& file) {
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

void applyNoise(std::vector<Eigen::Vector2d>& pointcloud, double amplitude) {
  std::uniform_real_distribution<double> dist(-amplitude, amplitude);
  std::mt19937 engine;
  for (auto& point : pointcloud) {
    point[0] += dist(engine);
    point[1] += dist(engine);
  }
}