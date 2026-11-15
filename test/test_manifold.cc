#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "manifold_models.hh"
#include "moptim/AnalyticalCost.hh"
#include "moptim/GaussNewton.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/Manifold.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/Observer.hh"
#include "test_models.hh"

using namespace moptim;
using namespace manifold_models;
using test_models::SimpleModel;
using test_models::TestData;

namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;

// ─── Data ─────────────────────────────────────────────────────────────────────────────────────

struct PlaneData {
  std::vector<double> inputs;  // [p, n] per element
  std::vector<double> offsets;
  Vec12 truth;
};

Vec6 tangent(double a, double b, double c, double d, double e, double f) {
  Vec6 v;
  v << a, b, c, d, e, f;
  return v;
}

// d = n . (T_true p), no noise.
PlaneData makePlaneData(size_t n, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> box(-5.0, 5.0);
  std::normal_distribution<double> gauss(0.0, 1.0);

  PlaneData data;
  data.truth = poseFromTangent(tangent(1.0, -0.5, 0.2, 0.3, -0.4, 0.5));
  const Eigen::Map<const Eigen::Matrix3d> R(data.truth.data());
  const Eigen::Map<const Eigen::Vector3d> t(data.truth.data() + 9);

  data.inputs.resize(6 * n);
  data.offsets.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d p(box(rng), box(rng), box(rng));
    const Eigen::Vector3d normal = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
    Eigen::Map<Eigen::Vector3d>{&data.inputs[6 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&data.inputs[6 * i + 3]} = normal;
    data.offsets[i] = normal.dot(R * p + t);
  }
  return data;
}

// An evaluation point away from the truth, so that the residuals are not zero.
Vec12 awayPose() { return poseFromTangent(tangent(0.4, 0.3, -0.2, 0.2, 0.1, -0.15)); }

struct LinearSystem {
  Eigen::MatrixXd JTJ;
  Eigen::VectorXd JTb;
  double cost = 0.0;
};

LinearSystem linearSystem(ICost<double>& cost, const double* x) {
  LinearSystem system;
  // Poison the buffers: they must be overwritten, not accumulated into.
  system.JTJ = Eigen::MatrixXd::Constant(cost.tangentDim(), cost.tangentDim(), 123.0);
  system.JTb = Eigen::VectorXd::Constant(cost.tangentDim(), 123.0);
  cost.computeLinearSystem(x, system.JTJ.data(), system.JTb.data(), system.cost);
  return system;
}

double relativeDifference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).norm() / std::max(b.norm(), 1e-300);
}

std::shared_ptr<const IManifold<double>> se3() { return std::make_shared<SE3Manifold>(); }

// ─── 1. A Euclidean manifold is no manifold ───────────────────────────────────────────────────

template <class Optimizer, class Cost>
void expectSameRun(Optimizer& plain, Optimizer& through_manifold, const std::shared_ptr<Cost>& plain_cost,
                   const std::shared_ptr<Cost>& manifold_cost) {
  plain.addCost(plain_cost);
  through_manifold.addCost(manifold_cost);

  Eigen::Vector2d x_plain{0.9, 0.2};
  Eigen::Vector2d x_manifold = x_plain;
  const Result<double> expected = plain.optimize(x_plain.data());
  const Result<double> actual = through_manifold.optimize(x_manifold.data());

  EXPECT_GT(expected.iterations, 1u);
  EXPECT_EQ(actual.iterations, expected.iterations);
  EXPECT_EQ(actual.final_cost, expected.final_cost);
  EXPECT_EQ(actual.status, expected.status);
  EXPECT_EQ(x_manifold[0], x_plain[0]);
  EXPECT_EQ(x_manifold[1], x_plain[1]);
}

using SimpleCost = AnalyticalCost<SimpleModel<double>, double>;

std::shared_ptr<SimpleCost> simpleCost(size_t param_dim) {
  return std::make_shared<SimpleCost>(TestData<double>::x_data_, TestData<double>::y_data_,
                                      TestData<double>::num_measurements, 1, 1, param_dim);
}

std::shared_ptr<SimpleCost> simpleCost(std::shared_ptr<const IManifold<double>> manifold) {
  return std::make_shared<SimpleCost>(TestData<double>::x_data_, TestData<double>::y_data_,
                                      TestData<double>::num_measurements, 1, 1, std::move(manifold));
}

}  // namespace

TEST(Manifold, EuclideanManifoldEqualsNoManifoldLevenbergMarquardt) {
  const auto manifold = std::make_shared<EuclideanManifold<double>>(2);
  LevenbergMarquardt<double> plain(2);
  LevenbergMarquardt<double> through_manifold(manifold);
  expectSameRun(plain, through_manifold, simpleCost(2), simpleCost(manifold));
}

TEST(Manifold, EuclideanManifoldEqualsNoManifoldGaussNewton) {
  const auto manifold = std::make_shared<EuclideanManifold<double>>(2);
  GaussNewton<double> plain(2);
  GaussNewton<double> through_manifold(manifold);
  expectSameRun(plain, through_manifold, simpleCost(2), simpleCost(manifold));
}

TEST(Manifold, EuclideanManifoldEqualsNoManifoldLinearSystem) {
  const auto manifold = std::make_shared<EuclideanManifold<double>>(2);
  const double x[2] = {0.9, 0.2};

  const auto check = [&](ICost<double>& plain, ICost<double>& through_manifold) {
    EXPECT_EQ(plain.manifold(), nullptr);
    EXPECT_EQ(through_manifold.manifold(), manifold.get());
    const LinearSystem expected = linearSystem(plain, x);
    const LinearSystem actual = linearSystem(through_manifold, x);
    EXPECT_EQ(actual.JTJ, expected.JTJ);
    EXPECT_EQ(actual.JTb, expected.JTb);
    EXPECT_EQ(actual.cost, expected.cost);
  };

  const auto* in = TestData<double>::x_data_;
  const auto* out = TestData<double>::y_data_;
  const size_t n = TestData<double>::num_measurements;
  {
    AnalyticalCost<SimpleModel<double>, double> plain(in, out, n, 1, 1, 2);
    AnalyticalCost<SimpleModel<double>, double> through_manifold(in, out, n, 1, 1, manifold);
    check(plain, through_manifold);
  }
  {
    NumericalCostForwardEuler<SimpleModel<double>, double> plain(in, out, n, 1, 1, 2);
    NumericalCostForwardEuler<SimpleModel<double>, double> through_manifold(in, out, n, 1, 1, manifold);
    check(plain, through_manifold);
  }
  {
    NumericalCostCentral<SimpleModel<double>, double> plain(in, out, n, 1, 1, 2);
    NumericalCostCentral<SimpleModel<double>, double> through_manifold(in, out, n, 1, 1, manifold);
    check(plain, through_manifold);
  }
}

// ─── 2. Ambient and tangent sizes differ: the circle ──────────────────────────────────────────

namespace {

template <class T>
std::vector<T> circleData(std::vector<T>& rotated, T angle) {
  const std::vector<T> points{1, 0, 0, 2, -1.5, 0.5, 0.7, -1.1, 2, 2};
  for (size_t i = 0; i < points.size(); i += 2) {
    rotated.push_back(std::cos(angle) * points[i] - std::sin(angle) * points[i + 1]);
    rotated.push_back(std::sin(angle) * points[i] + std::cos(angle) * points[i + 1]);
  }
  return points;
}

}  // namespace

template <class Optimizer>
class ManifoldOptimizers : public ::testing::Test {};
using Optimizers = ::testing::Types<LevenbergMarquardt<double>, GaussNewton<double>>;
TYPED_TEST_SUITE(ManifoldOptimizers, Optimizers);

TYPED_TEST(ManifoldOptimizers, FitsAnAngleOnTheCircle) {
  const double angle = 0.9;
  std::vector<double> rotated;
  const std::vector<double> points = circleData<double>(rotated, angle);
  const size_t n = points.size() / 2;

  const auto circle = std::make_shared<CircleManifold<double>>();
  auto cost = std::make_shared<AnalyticalCost<CircleRotationModel<double>, double>>(points.data(), rotated.data(), n, 2,
                                                                                    2, circle);
  EXPECT_EQ(cost->paramDim(), 2u);
  EXPECT_EQ(cost->tangentDim(), 1u);

  TypeParam solver(circle);
  EXPECT_EQ(solver.ambientDim(), 2u);
  EXPECT_EQ(solver.tangentDim(), 1u);
  solver.addCost(cost);
  solver.setMaxIterations(50);

  // The true angle is 2.5 rad away from the start.
  const double start = angle - 2.5;
  double x[2] = {std::cos(start), std::sin(start)};

  // The linear system is 1x1.
  const LinearSystem system = linearSystem(*cost, x);
  EXPECT_EQ(system.JTJ.rows(), 1);
  EXPECT_EQ(system.JTJ.cols(), 1);
  EXPECT_GT(system.JTJ(0, 0), 0.0);

  const Result<double> result = solver.optimize(x);
  EXPECT_NE(result.status, Status::NUMERIC_ERROR);
  EXPECT_NEAR(x[0] * x[0] + x[1] * x[1], 1.0, 1e-12);
  EXPECT_NEAR(std::atan2(x[1], x[0]), angle, 1e-7);
  EXPECT_LT(result.final_cost, 1e-12);
}

// ─── 3. SE(3) ─────────────────────────────────────────────────────────────────────────────────

TEST(Manifold, SE3PlusIsALeftIncrement) {
  const SE3Manifold manifold;
  const Vec12 x = awayPose();
  const Vec6 delta = tangent(0.1, -0.2, 0.3, 0.05, 0.1, -0.07);

  Vec12 moved;
  manifold.plus(x.data(), delta.data(), moved.data());

  Eigen::Matrix3d dR;
  Eigen::Vector3d dt;
  se3Exp(delta.data(), dR, dt);
  const Eigen::Matrix3d R = Eigen::Map<const Eigen::Matrix3d>(x.data());
  const Eigen::Vector3d t = Eigen::Map<const Eigen::Vector3d>(x.data() + 9);
  EXPECT_LT((Eigen::Map<const Eigen::Matrix3d>(moved.data()) - dR * R).norm(), 1e-15);
  EXPECT_LT((Eigen::Map<const Eigen::Vector3d>(moved.data() + 9) - (dR * t + dt)).norm(), 1e-15);
  EXPECT_LT((dR.transpose() * dR - Eigen::Matrix3d::Identity()).norm(), 1e-14);

  Vec12 unmoved;
  const Vec6 zero = Vec6::Zero();
  manifold.plus(x.data(), zero.data(), unmoved.data());
  EXPECT_EQ(unmoved, x);
}

TEST(Manifold, SE3AnalyticalMatchesNumericalThroughTheManifold) {
  const size_t n = 200;
  const PlaneData data = makePlaneData(n);
  const Vec12 x = awayPose();
  const auto manifold = se3();

  AnalyticalCost<PointToPlane, double> analytical(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
  NumericalCostCentral<PointToPlane, double> central(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
  NumericalCostForwardEuler<PointToPlane, double> forward(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);

  const LinearSystem expected = linearSystem(analytical, x.data());
  const LinearSystem numeric = linearSystem(central, x.data());
  EXPECT_EQ(expected.JTJ.rows(), 6);
  EXPECT_GT(expected.cost, 1.0);
  EXPECT_EQ(numeric.cost, expected.cost);
  EXPECT_LT(relativeDifference(numeric.JTJ, expected.JTJ), 1e-6);
  EXPECT_LT(relativeDifference(numeric.JTb, expected.JTb), 1e-6);

  const LinearSystem one_sided = linearSystem(forward, x.data());
  EXPECT_LT(relativeDifference(one_sided.JTJ, expected.JTJ), 1e-5);
  EXPECT_LT(relativeDifference(one_sided.JTb, expected.JTb), 1e-5);
}

TEST(Manifold, SE3PointToPointAnalyticalMatchesNumerical) {
  const size_t n = 100;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> box(-5.0, 5.0);
  const Vec12 truth = poseFromTangent(tangent(1.0, -0.5, 0.2, 0.3, -0.4, 0.5));
  std::vector<double> points(3 * n), targets(3 * n);
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d p(box(rng), box(rng), box(rng));
    Eigen::Map<Eigen::Vector3d>{&points[3 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&targets[3 * i]} =
        Eigen::Map<const Eigen::Matrix3d>(truth.data()) * p + Eigen::Map<const Eigen::Vector3d>(truth.data() + 9);
  }
  const Vec12 x = awayPose();
  const auto manifold = se3();

  AnalyticalCost<PointToPoint, double> analytical(points.data(), targets.data(), n, 3, 3, manifold);
  NumericalCostCentral<PointToPoint, double> central(points.data(), targets.data(), n, 3, 3, manifold);
  const LinearSystem expected = linearSystem(analytical, x.data());
  const LinearSystem numeric = linearSystem(central, x.data());
  EXPECT_LT(relativeDifference(numeric.JTJ, expected.JTJ), 1e-6);
  EXPECT_LT(relativeDifference(numeric.JTb, expected.JTb), 1e-6);
}

TYPED_TEST(ManifoldOptimizers, RecoversAKnownPoseFromAFarStart) {
  const size_t n = 300;
  const PlaneData data = makePlaneData(n);
  const auto manifold = se3();

  auto cost = std::make_shared<AnalyticalCost<PointToPlane, double>>(data.inputs.data(), data.offsets.data(), n, 6, 1,
                                                                     manifold);
  TypeParam solver(manifold);
  solver.addCost(cost);
  solver.setMaxIterations(50);

  Vec12 x = poseFromTangent(tangent(-1.0, 2.0, 0.5, -0.6, 0.5, 0.4));
  const Result<double> result = solver.optimize(x.data());

  EXPECT_NE(result.status, Status::NUMERIC_ERROR);
  EXPECT_LT(result.final_cost, 1e-16);
  EXPECT_LT((x - data.truth).norm(), 1e-7);
  // Still a rotation matrix.
  const Eigen::Matrix3d R = Eigen::Map<const Eigen::Matrix3d>(x.data());
  EXPECT_LT((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
}

// ─── 4. Convention guard ──────────────────────────────────────────────────────────────────────

TEST(Manifold, AJacobianOfTheWrongIncrementSideIsDetected) {
  const size_t n = 200;
  const PlaneData data = makePlaneData(n);
  const Vec12 x = awayPose();
  const auto manifold = se3();

  NumericalCostCentral<PointToPlane, double> numeric(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
  AnalyticalCost<PointToPlane, double> left(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
  AnalyticalCost<PointToPlaneRightIncrement, double> right(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);

  const LinearSystem expected = linearSystem(numeric, x.data());
  const double left_error = relativeDifference(linearSystem(left, x.data()).JTb, expected.JTb);
  const double right_error = relativeDifference(linearSystem(right, x.data()).JTb, expected.JTb);

  EXPECT_LT(left_error, 1e-6);
  EXPECT_GT(right_error, 1e-2);
  EXPECT_GT(right_error, 1e4 * left_error);
}

// ─── 5. Product manifold ──────────────────────────────────────────────────────────────────────

namespace {

// Residual obs - x[12 + k], for the Euclidean block of an SE(3) x R^2 product.
struct OffsetAnchor {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* /*input*/, const double* obs, double* res) const {
    res[0] = obs[0] - x[12];
    res[1] = obs[1] - x[13];
  }
  // Derivative with respect to all 8 tangent dimensions: only the last two are non-zero.
  void jacobian(const double* /*x*/, const double* /*input*/, const double* /*obs*/, double* jac) const {
    for (int i = 0; i < 16; ++i) jac[i] = 0.0;
    jac[6 + 8 * 0] = -1.0;
    jac[7 + 8 * 1] = -1.0;
  }
};

// The point-to-plane model reading the pose from the first 12 entries of a larger x.
using PlaneInProduct = PointToPlane;

std::shared_ptr<const IManifold<double>> poseAndOffset() {
  return std::make_shared<ProductManifold<double>>(std::vector<std::shared_ptr<const IManifold<double>>>{
      std::make_shared<SE3Manifold>(), std::make_shared<EuclideanManifold<double>>(2)});
}

}  // namespace

TEST(ProductManifold, SizesAndPlus) {
  const auto product = poseAndOffset();
  EXPECT_EQ(product->ambientDim(), 14u);
  EXPECT_EQ(product->tangentDim(), 8u);

  Eigen::Matrix<double, 14, 1> x;
  x << awayPose(), 1.5, -2.5;
  Eigen::Matrix<double, 8, 1> delta;
  delta << tangent(0.1, -0.2, 0.3, 0.05, 0.1, -0.07), 0.25, 4.0;

  Eigen::Matrix<double, 14, 1> moved;
  product->plus(x.data(), delta.data(), moved.data());

  Vec12 pose;
  SE3Manifold().plus(x.data(), delta.data(), pose.data());
  EXPECT_EQ(Vec12(moved.head<12>()), pose);
  EXPECT_EQ(moved[12], 1.75);
  EXPECT_EQ(moved[13], 1.5);

  Eigen::Matrix<double, 8, 1> zero = Eigen::Matrix<double, 8, 1>::Zero();
  product->plus(x.data(), zero.data(), moved.data());
  EXPECT_EQ(moved, x);
}

TEST(ProductManifold, ACostOnTheLeadingBlockLeavesTheRestOfTheSystemZero) {
  const size_t n = 50;
  const PlaneData data = makePlaneData(n);
  const auto product = poseAndOffset();
  Eigen::Matrix<double, 14, 1> x;
  x << awayPose(), 1.5, -2.5;

  const auto check = [&](ICost<double>& cost) {
    EXPECT_EQ(cost.activeParamDim(), 6u);
    const LinearSystem system = linearSystem(cost, x.data());
    ASSERT_EQ(system.JTJ.rows(), 8);
    EXPECT_GT(system.JTJ.topLeftCorner(6, 6).norm(), 1.0);
    EXPECT_EQ(system.JTJ.topRightCorner(6, 2).norm(), 0.0);
    EXPECT_EQ(system.JTJ.bottomRows(2).norm(), 0.0);
    EXPECT_GT(system.JTb.head(6).norm(), 1.0);
    EXPECT_EQ(system.JTb.tail(2).norm(), 0.0);
  };

  AnalyticalCost<PlaneInProduct, double> analytical(data.inputs.data(), data.offsets.data(), n, 6, 1, product,
                                                    PlaneInProduct{}, 6);
  NumericalCostCentral<PlaneInProduct, double> central(data.inputs.data(), data.offsets.data(), n, 6, 1, product,
                                                       PlaneInProduct{}, 6);
  check(analytical);
  check(central);
}

TEST(ProductManifold, OptimizesWithCostsOnTheLeadingBlockAndOnAllDimensions) {
  const size_t n = 200;
  const PlaneData data = makePlaneData(n);
  const auto product = poseAndOffset();

  const double anchor_input = 0.0;
  const double anchor_observation[2] = {3.0, -1.0};

  LevenbergMarquardt<double> solver(product);
  solver.addCost(std::make_shared<AnalyticalCost<PlaneInProduct, double>>(data.inputs.data(), data.offsets.data(), n, 6,
                                                                          1, product, PlaneInProduct{}, 6));
  solver.addCost(
      std::make_shared<AnalyticalCost<OffsetAnchor, double>>(&anchor_input, anchor_observation, 1, 1, 2, product));
  solver.setMaxIterations(50);

  Eigen::Matrix<double, 14, 1> x;
  x << poseFromTangent(tangent(-1.0, 2.0, 0.5, -0.6, 0.5, 0.4)), 0.0, 0.0;
  const Result<double> result = solver.optimize(x.data());

  EXPECT_LT(result.final_cost, 1e-16);
  EXPECT_LT((Vec12(x.head<12>()) - data.truth).norm(), 1e-7);
  EXPECT_NEAR(x[12], 3.0, 1e-8);
  EXPECT_NEAR(x[13], -1.0, 1e-8);
}

// ─── 6. A rejected trial leaves x untouched ───────────────────────────────────────────────────

namespace {

// The same linear system wherever it is asked, and a cost that is worse than it at every other point.
class WorseEverywhereCost : public ICost<double> {
 public:
  explicit WorseEverywhereCost(std::shared_ptr<const IManifold<double>> manifold)
      : ICost<double>(1, 1, std::move(manifold), 1) {}

  double computeCost(const double* /*x*/) override { return 5.0; }

  void computeLinearSystem(const double* /*x*/, double* JTJ, double* JTb, double& cost) override {
    JTJ[0] = 1.0;
    JTb[0] = -1.0;
    cost = 1.0;
  }
};

class CountingCircle : public CircleManifold<double> {
 public:
  void plus(const double* x, const double* delta, double* x_plus) const override {
    ++plus_calls;
    CircleManifold<double>::plus(x, delta, x_plus);
  }
  mutable int plus_calls = 0;
};

class PhaseRecorder : public IOptimizerObserver<double> {
 public:
  void onIteration(const IterationEvent<double>& event) override { phases.push_back(event.phase); }
  void onLinearSystem(const LinearSystemEvent<double>& /*event*/) override {}
  std::vector<Phase> phases;
};

}  // namespace

TEST(Manifold, ARejectedLevenbergMarquardtTrialLeavesXBitForBitUntouched) {
  const auto circle = std::make_shared<CountingCircle>();
  LevenbergMarquardt<double> solver(circle);
  solver.addCost(std::make_shared<WorseEverywhereCost>(circle));
  PhaseRecorder recorder;
  solver.setObserver(&recorder);

  double x[2] = {0.6, 0.8};
  const double before[2] = {x[0], x[1]};
  const Status status = solver.step(x);

  EXPECT_EQ(status, Status::STEP_OK);
  ASSERT_FALSE(recorder.phases.empty());
  for (const Phase phase : recorder.phases) {
    EXPECT_EQ(phase, Phase::REJECTED);
  }
  EXPECT_GT(circle->plus_calls, 0);  // The trial points were really computed.
  EXPECT_EQ(std::memcmp(x, before, sizeof(x)), 0);
}

// ─── 7. Numerical cost: single and multiple chunks ────────────────────────────────────────────

TEST(Manifold, NumericalCostThroughTheManifoldInSingleAndMultipleChunks) {
  const auto manifold = se3();
  const Vec12 x = awayPose();

  // One chunk holds about 16 KiB / (7 doubles) = 292 elements of this problem.
  for (const size_t n : {size_t{40}, size_t{3000}}) {
    SCOPED_TRACE(n);
    const PlaneData data = makePlaneData(n, 9);
    AnalyticalCost<PointToPlane, double> analytical(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
    NumericalCostCentral<PointToPlane, double> central(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);
    NumericalCostForwardEuler<PointToPlane, double> forward(data.inputs.data(), data.offsets.data(), n, 6, 1, manifold);

    const LinearSystem expected = linearSystem(analytical, x.data());
    const LinearSystem numeric = linearSystem(central, x.data());
    EXPECT_LT(relativeDifference(numeric.JTJ, expected.JTJ), 1e-6);
    EXPECT_LT(relativeDifference(numeric.JTb, expected.JTb), 1e-6);
    const LinearSystem one_sided = linearSystem(forward, x.data());
    EXPECT_LT(relativeDifference(one_sided.JTJ, expected.JTJ), 1e-5);
    EXPECT_LT(relativeDifference(one_sided.JTb, expected.JTb), 1e-5);

    // The point table is rebuilt at every linear system: a second call at another x agrees too.
    const Vec12 other = poseFromTangent(tangent(-0.2, 0.1, 0.4, -0.1, 0.2, 0.3));
    const LinearSystem expected_other = linearSystem(analytical, other.data());
    const LinearSystem numeric_other = linearSystem(central, other.data());
    EXPECT_LT(relativeDifference(numeric_other.JTb, expected_other.JTb), 1e-6);
  }
}

// ─── 8. float ─────────────────────────────────────────────────────────────────────────────────

TEST(Manifold, OptimizesOnTheCircleInSinglePrecision) {
  const float angle = 0.9f;
  std::vector<float> rotated;
  const std::vector<float> points = circleData<float>(rotated, angle);
  const size_t n = points.size() / 2;

  const auto circle = std::make_shared<CircleManifold<float>>();
  auto cost = std::make_shared<AnalyticalCost<CircleRotationModel<float>, float>>(points.data(), rotated.data(), n, 2,
                                                                                  2, circle);
  LevenbergMarquardt<float> solver(circle);
  solver.addCost(cost);
  solver.setMaxIterations(50);

  const float start = angle - 2.5f;
  float x[2] = {std::cos(start), std::sin(start)};
  solver.optimize(x);

  EXPECT_NEAR(x[0] * x[0] + x[1] * x[1], 1.0f, 1e-5f);
  EXPECT_NEAR(std::atan2(x[1], x[0]), angle, 1e-3f);
}
