#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief Generates a 3D point cloud
 *
 * @param n_points number of points to generate
 * @return std::vector<Eigen::Vector3d>
 */
std::vector<Eigen::Vector3d> generateCloud(size_t n_points) {
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