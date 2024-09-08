#include <gtest/gtest.h>

#include <Cost.hh>
#include <GaussNewton.hh>

#include "ConsoleLogger.hh"

struct Pt2Dist {
  Pt2Dist(const Eigen::VectorXd& x) {
    transform_.setIdentity();
    transform_.rotate(x[2]);
    transform_.translate(Eigen::Vector2d{x[0], x[1]});
  }

  Eigen::Vector2d operator()(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    return target - transform_ * source;
  }

  Eigen::Affine2d transform_;
};
Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Affine2d& transform) {
  return transform * point;
}

const std::vector<Eigen::Vector2d> pointcloud{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

class Test2DTransform : public ::testing::Test {
 protected:
};

TEST(Test2DTransform, test_simple) {
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};

  Eigen::Rotation2D<double> rot(x0_ref[2]);
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();
  transform.translate(Eigen::Vector2d{x0_ref[0], x0_ref[1]});
  transform.rotate(rot);

  std::vector<Eigen::Vector2d> transformed_pointcloud;

  std::transform(pointcloud.begin(), pointcloud.end(), std::back_inserter(transformed_pointcloud),
                 [&](const Eigen::Vector2d& pt) { return transformPoint(pt, transform); });

  GaussNewton solver(std::make_shared<ConsoleLogger>());
  auto cost =
      std::make_shared<Cost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud, &pointcloud, 3);

  solver.addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver.optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}