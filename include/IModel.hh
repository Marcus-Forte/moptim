#pragma once

#include <Eigen/Dense>

class IModel {
 public:
  IModel(const Eigen::VectorXd& x0) : x_(x0){};

 protected:
  const Eigen::VectorXd x_;
};