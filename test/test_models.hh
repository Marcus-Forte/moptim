#pragma once
#include "IModel.hh"

struct Point2 {
    double x;
    double y;
    
    Point2 operator-(const Point2& rhs) const {
      return {x - rhs.x, y - rhs.y};
    }

    Point2 operator/(double rhs) const {
      return {x / rhs, y / rhs};
    }
  };
  

struct SimpleModel : public IModel {
    // State (x)
  SimpleModel(const Eigen::VectorXd& x0) : IModel(x0) {}

  // Error function
  double operator()(double input, double measurement) const;
};


// parameter vector `x` represents x, y, theta.
struct Point2Distance : public IModel {
  public:

  Point2Distance(const Eigen::VectorXd& x0) : IModel(x0) {
    T_ << std::cos(x_[2]), -std::sin(x_[2]), x_[0], std::sin(x_[2]), std::cos(x_[2]), x_[1], 0, 0, 1;
  }

  Point2 operator()(const Point2& point,const Point2& measurement) const;

  private:
  Eigen::Matrix3d T_;
  };