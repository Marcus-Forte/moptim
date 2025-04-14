#include <gtest/gtest.h>

#include <GaussNewton.hh>
#include <NumericalCost.hh>

#include "ConsoleLogger.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "data/generate.hh"

namespace {

constexpr size_t num_points_test = 50000000;

std::shared_ptr<ConsoleLogger> g_logging;

Eigen::Vector3d transformPoint(const Eigen::Vector3d& point, const Eigen::Affine3d& transform) {
  return transform * point;
}

}  // namespace

struct Pt3Dist {
  Pt3Dist(const Eigen::VectorXd& x) : x_(x) {
    transform_.setIdentity();
    // rotate using three angles, x3, x4, x4
    Eigen::AngleAxisd rollAngle(x_[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(x_[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(x_[5], Eigen::Vector3d::UnitZ());
    transform_.rotate(rollAngle * pitchAngle * yawAngle);
    // translate using x0, x1, x2
    transform_.translate(Eigen::Vector3d{x_[0], x_[1], x_[2]});
  }

  Eigen::Vector3d operator()(const Eigen::Vector3d& source, const Eigen::Vector3d& target) const {
    return target - transformPoint(source, transform_);
  }

  Eigen::Affine3d transform_;
  Eigen::Vector<double, 6> x_;
};

class Test3DTransform : public ::testing::Test {
 public:
  void SetUp() override {
    g_logging = std::make_shared<ConsoleLogger>();
    g_logging->setLevel(ILog::Level::DEBUG);
    pointcloud_ = generateCloud(num_points_test);

    Eigen::AngleAxisd rollAngle(x0_ref[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(x0_ref[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(x0_ref[5], Eigen::Vector3d::UnitZ());

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();

    transform.translate(Eigen::Vector3d{x0_ref[0], x0_ref[1], x0_ref[2]});
    transform.rotate(rollAngle * pitchAngle * yawAngle);
    std::transform(pointcloud_.begin(), pointcloud_.end(), std::back_inserter(transformed_pointcloud_),
                   [&](const Eigen::Vector3d& pt) { return transformPoint(pt, transform); });
  }

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3, 0, 0, 0}};
  std::vector<Eigen::Vector3d> transformed_pointcloud_;
  std::vector<Eigen::Vector3d> pointcloud_;
  std::shared_ptr<IOptimizer> solver_;
};

TEST_F(Test3DTransform, SyclCost) {
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};

  auto cost = std::make_shared<NumericalCostSycl<Eigen::Vector3d, Eigen::Vector3d, Pt3Dist>>(&transformed_pointcloud_,
                                                                                             &pointcloud_, queue);
  auto known_cost = std::make_shared<NumericalCost<Eigen::Vector3d, Eigen::Vector3d, Pt3Dist>>(&transformed_pointcloud_,
                                                                                               &pointcloud_);
  Eigen::VectorXd x0{{0.0, 0.0, 0, 0, 0, 0}};

  t0.start();
  const auto known_cost_sum = known_cost->computeCost(x0);
  auto stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Known cost: {} took {} us", known_cost_sum, stop);

  t0.start();
  const auto cost_sum = cost->computeCost(x0);
  stop = t0.stop();
  g_logging->log(ILog::Level::INFO, "Sycl cost: {} took {} us", cost_sum, stop);

  EXPECT_NEAR(cost_sum, known_cost_sum, 1e-5);
}
