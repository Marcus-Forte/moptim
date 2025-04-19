#pragma once

#include <Eigen/Dense>

void compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance = 1e-5);