#pragma once

#include <Eigen/Dense>
#include <filesystem>

void compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance = 1e-5);
std::vector<Eigen::Vector2d> read2DTxtScan(std::filesystem::path&& file);
void applyNoise(std::vector<Eigen::Vector2d>& pointcloud, double amplitude);