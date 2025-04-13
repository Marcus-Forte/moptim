#include <gtest/gtest.h>

#include <GaussNewton.hh>
#include <NumericalCost.hh>
#include <random>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "data/parser.hh"

std::shared_ptr<ConsoleLogger> g_logging;

namespace {

Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Affine2d& transform) {
  return transform * point;
}

void applyNoise(std::vector<Eigen::Vector2d>& pointcloud, double amplitude) {
  std::uniform_real_distribution<double> dist(-amplitude, amplitude);
  std::mt19937 engine;
  for (auto& point : pointcloud) {
    point[0] += dist(engine);
    point[1] += dist(engine);
  }
}

}  // namespace

struct Pt2Dist {
  Pt2Dist(const Eigen::VectorXd& x) : x_(x) {
    transform_.setIdentity();
    transform_.rotate(x_[2]);
    transform_.translate(Eigen::Vector2d{x_[0], x_[1]});
  }

  Eigen::Vector2d operator()(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    return target - transformPoint(source, transform_);
  }

  Eigen::Matrix<double, 2, 3> jacobian(const Eigen::Vector2d& source, const Eigen::Vector2d& target) const {
    Eigen::Matrix<double, 2, 3> jac;
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

class Test2DTransform : public ::testing::Test {
 public:
  void SetUp() override {
    g_logging = std::make_shared<ConsoleLogger>();
    g_logging->setLevel(ILog::Level::DEBUG);
    pointcloud_ = read2DTxtScan(TEST_PATH / std::filesystem::path("scan.txt"));

    Eigen::Rotation2D<double> rot(x0_ref[2]);
    Eigen::Affine2d transform = Eigen::Affine2d::Identity();

    transform.translate(Eigen::Vector2d{x0_ref[0], x0_ref[1]});
    transform.rotate(rot);
    std::transform(pointcloud_.begin(), pointcloud_.end(), std::back_inserter(transformed_pointcloud_),
                   [&](const Eigen::Vector2d& pt) { return transformPoint(pt, transform); });
    ::applyNoise(transformed_pointcloud_, 0.01);
  }

 protected:
  Eigen::VectorXd x0_ref{{0.1, 0.2, 0.3}};
  std::vector<Eigen::Vector2d> transformed_pointcloud_;
  std::vector<Eigen::Vector2d> pointcloud_;
  std::shared_ptr<IOptimizer> solver_;
};

TEST_F(Test2DTransform, SyclCost) {
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};
  t0.start();
  auto cost = std::make_shared<NumericalCostSycl<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                             &pointcloud_, queue);
  auto known_cost = std::make_shared<NumericalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                               &pointcloud_);
  Eigen::VectorXd x0{{0.1, 0.1, 0}};
  const auto cost_sum = cost->computeCost(x0);
  const auto known_cost_sum = known_cost->computeCost(x0);

  EXPECT_NEAR(cost_sum, known_cost_sum, 1e-5);
  g_logging->log(ILog::Level::INFO, "Sum: {}", known_cost_sum);
}

TEST_F(Test2DTransform, 2DTransformGN) {
  Timer t0;
  sycl::queue queue{sycl::default_selector_v};
  t0.start();
  solver_ = std::make_shared<GaussNewton>(g_logging);
  g_logging->log(ILog::Level::INFO, "Sycl Device: prepare");
  auto cost = std::make_shared<NumericalCostSycl<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                             &pointcloud_, queue);
  g_logging->log(ILog::Level::INFO, "Sycl Device: ready");
  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  auto delta = t0.stop();
  g_logging->log(ILog::Level::INFO, "GN Elapsed us: {}. Registered points: {}", delta, pointcloud_.size());
}

TEST_F(Test2DTransform, 2DTransformLM) {
  Timer t0;
  t0.start();
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  auto cost =
      std::make_shared<NumericalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist, DifferentiationMethod::BACKWARD_EULER>>(
          &transformed_pointcloud_, &pointcloud_);
  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  auto delta = t0.stop();
  g_logging->log(ILog::Level::INFO, "LM Elapsed us: {}. Registered points: {}", delta, pointcloud_.size());
}

// FIXME
TEST_F(Test2DTransform, DISABLED_2DTransformLMAnalytical) {
  solver_ = std::make_shared<LevenbergMarquardt>(g_logging);
  auto cost = std::make_shared<AnalyticalCost<Eigen::Vector2d, Eigen::Vector2d, Pt2Dist>>(&transformed_pointcloud_,
                                                                                          &pointcloud_);

  solver_->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver_->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
}
