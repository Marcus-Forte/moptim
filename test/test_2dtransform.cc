#include <gtest/gtest.h>

#include <GaussNewton.hh>
#include <NumericalCost.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"

struct Pt2Dist {
  Pt2Dist(const Eigen::VectorXd& x) {
    transform_.setIdentity();
    transform_.rotate(x[2]);
    transform_.translate(Eigen::Vector2d{x[0], x[1]});

    x_ = x;
  }

  Eigen::Vector2d operator()(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    return target - transform_ * source;
  }

  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian(const Eigen::Vector2d& source,
                                                        const Eigen::Vector2d& target) const {
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jac;
    const auto cos_theta = std::cos(x_[2]);
    const auto sin_theta = std::sin(x_[2]);
    jac(0, 0) = -1;
    jac(0, 1) = 0;
    jac(0, 2) = -cos_theta * source[0] + sin_theta * source[1];

    jac(1, 0) = 0;
    jac(1, 1) = -1;
    jac(1, 2) = sin_theta * source[0] + cos_theta * source[1];
    return jac;
  }

  Eigen::Affine2d transform_;
  Eigen::Vector3d x_;
};
Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Affine2d& transform) {
  return transform * point;
}

class Test2DTransform : public ::testing::Test {
 public:
  void SetUp() override {
    Eigen::Rotation2D<double> rot(x0_ref[2]);
    Eigen::Affine2d transform = Eigen::Affine2d::Identity();

    transform.translate(Eigen::Vector2d{x0_ref[0], x0_ref[1]});
    transform.rotate(rot);
    std::transform(pointcloud_.begin(), pointcloud_.end(), std::back_inserter(transformed_pointcloud_),
                   [&](const Eigen::Vector2d& pt) { return transformPoint(pt, transform); });
  }

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  std::shared_ptr<IOptimizer> solver_;
};

TEST_F(Test2DTransform, 2DTransform) {
  solver_ = std::make_shared<GaussNewton>(std::make_shared<ConsoleLogger>());
  auto cost = std::make_shared<NumericalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                         &pointcloud_);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}

TEST_F(Test2DTransform, 2DTransformLM) {
  solver_ = std::make_shared<LevenbergMarquardt>(std::make_shared<ConsoleLogger>());
  auto cost =
      std::make_shared<NumericalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist, DifferentiationMethod::CENTRAL>>(
          &transformed_pointcloud_, &pointcloud_);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}

// FIXME
TEST_F(Test2DTransform, DISABLED_2DTransformLMAnalytical) {
  solver_ = std::make_shared<LevenbergMarquardt>(std::make_shared<ConsoleLogger>());
  auto cost = std::make_shared<AnalyticalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                          &pointcloud_);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}
