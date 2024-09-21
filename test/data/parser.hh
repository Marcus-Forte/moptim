#pragma once

#include <vector>
#include <filesystem>
#include <fstream>
#include <Eigen/Dense>
#include <iostream>

/**
 * @brief Reads ASC point cloud. Assume coordinates are the first two columns. Delimiter is whitespace.
 * 
 * @param file 
 * @return std::vector<Eigen::Vector2d> 
 */
std::vector<Eigen::Vector2d> read2DTxtScan(std::filesystem::path&& file) {
    if(!std::filesystem::exists(file)) {
        throw  std::runtime_error("File does not exist: " + file.string());
    }
    std::ifstream pointcloud_file(file);

    double x;
    double y;
    double discard;
    std::vector<Eigen::Vector2d> pointcloud;
    std::string line;
   while(pointcloud_file >> x >> y >> discard >> discard >> discard >> discard >> discard >> discard >> discard) {
    pointcloud.emplace_back(x,y);
   }

    return pointcloud;
}