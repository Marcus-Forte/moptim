#include "test_models.hh"

double SimpleModel::operator()(double input, double measurement) const {
  return measurement - x_[0] * input / (x_[1] + input);
}

Point2 Point2Distance::operator()(const Point2& point, const Point2& measurement) const {
  Eigen::Vector3d p{{point.x, point.y, 1.0}};
  Eigen::Vector3d p_tf = T_ * p;

  return {measurement.x - p_tf[0], measurement.y - p_tf[1]};
}