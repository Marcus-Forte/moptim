// FusedCost: several costs in one launch. Every comparison with the costs run on their own is bitwise.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "manifold_models.hh"
#include "moptim/AnalyticalCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/FusedCost.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"

using namespace moptim;

namespace {

constexpr size_t kParams = 4;

// Two residuals and four parameters, nonlinear (sin), const-callable.
struct CurveModel {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] * in[0] + std::sin(x[1] * in[1]) + x[2]);
    res[1] = obs[1] - (x[3] * in[0] * in[1] + x[0] * x[1]);
  }
  void jacobian(const double* x, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -in[0];
    jac[1] = -in[1] * std::cos(x[1] * in[1]);
    jac[2] = -1.0;
    jac[3] = 0.0;
    jac[4] = -x[1];
    jac[5] = -x[0];
    jac[6] = 0.0;
    jac[7] = -in[0] * in[1];
  }
};

// A prior on the first two parameters (one element, like an IMU factor): non-const residual, so the
// work-items each get their own copy.
struct PriorModel {
  void setState(const double* x) { offset_ = 0.25 * x[3]; }
  void residual(const double* x, const double* in, const double* /*obs*/, double* res) {
    res[0] = 3.0 * (x[0] - in[0]) + offset_;
    res[1] = 2.0 * (x[1] - in[1]) * (x[1] - in[1]);
  }
  double offset_ = 0.0;
};

struct Data {
  std::vector<double> input;
  std::vector<double> observations;
};

Data makeData(size_t n, const double* truth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::normal_distribution<double> noise(0.0, 0.05);
  Data data;
  CurveModel model;
  for (size_t e = 0; e < n; ++e) {
    const double in[2] = {value(rng), value(rng)};
    double obs[2];
    model.residual(truth, in, std::array<double, 2>{0.0, 0.0}.data(), obs);
    data.input.push_back(in[0]);
    data.input.push_back(in[1]);
    data.observations.push_back(-obs[0] + noise(rng));
    data.observations.push_back(-obs[1] + noise(rng));
  }
  return data;
}

std::shared_ptr<ExecutionContext> makeContext(size_t threads) {
  return std::make_shared<ExecutionContext>(threads);
}

bool sameBits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

bool sameBits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

struct System {
  std::vector<double> jtj = std::vector<double>(kParams * kParams, 7.0);
  std::vector<double> jtb = std::vector<double>(kParams, 7.0);
  double cost = 7.0;
};

const double kX[kParams] = {0.3, -0.7, 0.2, 1.1};
const double kTruth[kParams] = {0.5, -0.4, 0.1, 0.9};
const double kPrior[2] = {0.45, -0.35};

using Curve = AnalyticalCost<CurveModel, double>;
using CurveCentral = NumericalCostCentral<CurveModel, double>;
using Prior = NumericalCostForwardEuler<PriorModel, double>;

// The three kinds of cost of one problem, on `context`.
struct Problem {
  Problem(const Data& data, size_t n, const std::shared_ptr<ExecutionContext>& context)
      : curve(std::make_shared<Curve>(data.input.data(), data.observations.data(), n, 2, 2, kParams, CurveModel{}, 0,
                                      context)),
        central(std::make_shared<CurveCentral>(data.input.data(), data.observations.data(), n / 3, 2, 2, kParams,
                                               CurveModel{}, 0, context)),
        prior(std::make_shared<Prior>(kPrior, kPrior, 1, 2, 2, kParams, PriorModel{}, 2, context)) {
    central->setLossFunction(std::make_shared<HuberLoss<double>>(0.5));
    central->setWeights(std::vector<double>(n / 3, 1.5));
  }
  std::shared_ptr<Curve> curve;
  std::shared_ptr<CurveCentral> central;
  std::shared_ptr<Prior> prior;
};

/// What IOptimizer computes from the costs one by one: zero, then each cost added in order.
System sumOfParts(const std::vector<ICost<double>*>& costs, const double* x) {
  System total;
  Eigen::Map<Eigen::MatrixXd> jtj(total.jtj.data(), kParams, kParams);
  Eigen::Map<Eigen::VectorXd> jtb(total.jtb.data(), kParams);
  jtj.setZero();
  jtb.setZero();
  total.cost = 0.0;
  for (ICost<double>* cost : costs) {
    System part;
    cost->computeLinearSystem(x, part.jtj.data(), part.jtb.data(), part.cost);
    jtj += Eigen::Map<const Eigen::MatrixXd>(part.jtj.data(), kParams, kParams);
    jtb += Eigen::Map<const Eigen::VectorXd>(part.jtb.data(), kParams);
    total.cost += part.cost;
  }
  return total;
}

}  // namespace

TEST(FusedCost, LinearSystemAndCostEqualTheSumOfItsCostsInOneLaunch) {
  const Data data = makeData(6000, kTruth, 5);
  const auto context = makeContext(0);
  Problem problem(data, 6000, context);
  const System expected = sumOfParts({problem.curve.get(), problem.central.get(), problem.prior.get()}, kX);
  double expected_cost = 0.0;
  for (ICost<double>* cost : std::vector<ICost<double>*>{problem.curve.get(), problem.central.get(), problem.prior.get()}) {
    expected_cost += cost->computeCost(kX);
  }

  const auto fused = makeFusedCost<double>(problem.curve, problem.central, problem.prior);
  EXPECT_EQ(fused->numElements(), 6000u + 2000u + 1u);
  EXPECT_EQ(fused->tangentDim(), kParams);

  System got;
  const uint64_t before = context->launches();
  fused->computeLinearSystem(kX, got.jtj.data(), got.jtb.data(), got.cost);
  EXPECT_EQ(context->launches() - before, 1u);
  EXPECT_TRUE(sameBits(got.jtj, expected.jtj));
  EXPECT_TRUE(sameBits(got.jtb, expected.jtb));
  EXPECT_TRUE(sameBits(got.cost, expected.cost));

  const uint64_t before_cost = context->launches();
  const double cost = fused->computeCost(kX);
  EXPECT_EQ(context->launches() - before_cost, 1u);
  EXPECT_TRUE(sameBits(cost, expected_cost));
  // The linear system's cost is the cost: speculative LM relies on it.
  EXPECT_TRUE(sameBits(cost, got.cost));

  // submit / wait, twice in a row (the second submit completes the first).
  System async1, async2;
  fused->submitLinearSystem(kX, async1.jtj.data(), async1.jtb.data(), &async1.cost);
  fused->submitLinearSystem(kX, async2.jtj.data(), async2.jtb.data(), &async2.cost);
  fused->wait();
  EXPECT_TRUE(sameBits(async1.jtj, expected.jtj));
  EXPECT_TRUE(sameBits(async2.jtj, expected.jtj));
  EXPECT_TRUE(sameBits(async2.cost, expected.cost));
}

TEST(FusedCost, OptimizerMatchesTheSeparateCostsWithFewerLaunches) {
  const Data data = makeData(4000, kTruth, 9);
  for (const bool speculative : {false, true}) {
    SCOPED_TRACE(testing::Message() << "speculative=" << speculative);
    const auto context = makeContext(0);
    Problem problem(data, 4000, context);

    LevenbergMarquardt<double> separate(kParams);
    separate.setSpeculativeLinearSystems(speculative);
    separate.addCost(problem.curve);
    separate.addCost(problem.central);
    separate.addCost(problem.prior);
    std::vector<double> x_separate(kX, kX + kParams);
    const uint64_t before_separate = context->launches();
    const Result<double> result_separate = separate.optimize(x_separate.data());
    const uint64_t launches_separate = context->launches() - before_separate;

    LevenbergMarquardt<double> fused(kParams);
    fused.setSpeculativeLinearSystems(speculative);
    fused.addCost(makeFusedCost<double>(problem.curve, problem.central, problem.prior));
    std::vector<double> x_fused(kX, kX + kParams);
    const uint64_t before_fused = context->launches();
    const Result<double> result_fused = fused.optimize(x_fused.data());
    const uint64_t launches_fused = context->launches() - before_fused;

    EXPECT_TRUE(sameBits(x_fused, x_separate));
    EXPECT_EQ(result_fused.iterations, result_separate.iterations);
    EXPECT_EQ(result_fused.status, result_separate.status);
    EXPECT_TRUE(sameBits(result_fused.final_cost, result_separate.final_cost));
    EXPECT_EQ(launches_fused * 3, launches_separate);
  }
}

TEST(FusedCost, IsBitIdenticalAtAnyThreadCount) {
  const Data data = makeData(20000, kTruth, 13);
  System reference;
  bool have_reference = false;
  for (const size_t threads : {size_t{1}, size_t{2}, size_t{5}, size_t{0}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    const auto context = makeContext(threads);
    Problem problem(data, 20000, context);
    const auto fused = makeFusedCost<double>(problem.curve, problem.central, problem.prior);
    System got;
    fused->computeLinearSystem(kX, got.jtj.data(), got.jtb.data(), got.cost);
    if (!have_reference) {
      reference = got;
      have_reference = true;
      continue;
    }
    EXPECT_TRUE(sameBits(got.jtj, reference.jtj));
    EXPECT_TRUE(sameBits(got.jtb, reference.jtb));
    EXPECT_TRUE(sameBits(got.cost, reference.cost));
  }
}

TEST(FusedCost, CostsWithoutElementsAreSkipped) {
  const Data data = makeData(500, kTruth, 17);
  const auto context = makeContext(0);
  const auto curve =
      std::make_shared<Curve>(data.input.data(), data.observations.data(), 500, 2, 2, kParams, CurveModel{}, 0, context);
  const auto empty =
      std::make_shared<Curve>(data.input.data(), data.observations.data(), 0, 2, 2, kParams, CurveModel{}, 0, context);
  const System expected = sumOfParts({curve.get(), empty.get()}, kX);

  const auto fused = makeFusedCost<double>(curve, empty);
  System got;
  fused->computeLinearSystem(kX, got.jtj.data(), got.jtb.data(), got.cost);
  EXPECT_TRUE(sameBits(got.jtj, expected.jtj));
  EXPECT_TRUE(sameBits(got.cost, expected.cost));

  const auto all_empty = makeFusedCost<double>(empty);
  const uint64_t before = context->launches();
  all_empty->computeLinearSystem(kX, got.jtj.data(), got.jtb.data(), got.cost);
  EXPECT_EQ(context->launches(), before);
  EXPECT_EQ(got.cost, 0.0);
  EXPECT_TRUE(Eigen::Map<Eigen::MatrixXd>(got.jtj.data(), kParams, kParams).isZero(0.0));
}

TEST(FusedCost, RejectsCostsOfDifferentContextsOrSpaces) {
  const Data data = makeData(100, kTruth, 19);
  const auto a =
      std::make_shared<Curve>(data.input.data(), data.observations.data(), 100, 2, 2, kParams, CurveModel{}, 0, makeContext(0));
  const auto b =
      std::make_shared<Curve>(data.input.data(), data.observations.data(), 100, 2, 2, kParams, CurveModel{}, 0, makeContext(0));
  EXPECT_THROW((FusedCost<double, Curve, Curve>(a, b)), std::invalid_argument);
  EXPECT_THROW((FusedCost<double, Curve, Curve>(a, nullptr)), std::invalid_argument);
}

TEST(FusedCost, WorksOnAManifold) {
  // Point-to-plane on SE(3) plus point-to-point, both analytical, on the same manifold object.
  std::mt19937 rng(29);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const size_t n = 1500;
  std::vector<double> planes, offsets, points, targets;
  for (size_t i = 0; i < n; ++i) {
    Eigen::Vector3d p(value(rng), value(rng), value(rng));
    Eigen::Vector3d normal = Eigen::Vector3d(value(rng), value(rng), value(rng)).normalized();
    planes.insert(planes.end(), {p.x(), p.y(), p.z(), normal.x(), normal.y(), normal.z()});
    offsets.push_back(normal.dot(p) + 0.01 * value(rng));
    points.insert(points.end(), {p.x(), p.y(), p.z()});
    targets.insert(targets.end(), {p.x() + 0.02, p.y() - 0.01, p.z() + 0.005});
  }
  const auto manifold = std::make_shared<manifold_models::SE3Manifold>();
  const auto context = makeContext(0);
  using Plane = AnalyticalCost<manifold_models::PointToPlane, double>;
  using Point = AnalyticalCost<manifold_models::PointToPoint, double>;
  const auto plane = std::make_shared<Plane>(planes.data(), offsets.data(), n, 6, 1, manifold,
                                             manifold_models::PointToPlane{}, 0, context);
  const auto point = std::make_shared<Point>(points.data(), targets.data(), n / 2, 3, 3, manifold,
                                             manifold_models::PointToPoint{}, 0, context);

  Eigen::Matrix<double, 6, 1> delta;
  delta << 0.01, -0.02, 0.005, 0.002, -0.001, 0.003;
  const Eigen::Matrix<double, 12, 1> start = manifold_models::poseFromTangent(delta);

  LevenbergMarquardt<double> separate(manifold);
  separate.addCost(plane);
  separate.addCost(point);
  Eigen::Matrix<double, 12, 1> x_separate = start;
  const auto result_separate = separate.optimize(x_separate.data());

  LevenbergMarquardt<double> fused(manifold);
  fused.addCost(makeFusedCost<double>(plane, point));
  Eigen::Matrix<double, 12, 1> x_fused = start;
  const auto result_fused = fused.optimize(x_fused.data());

  EXPECT_EQ(std::memcmp(x_fused.data(), x_separate.data(), sizeof(double) * 12), 0);
  EXPECT_EQ(result_fused.iterations, result_separate.iterations);
  EXPECT_TRUE(sameBits(result_fused.final_cost, result_separate.final_cost));
}

// ─── Reusing a cost: setData() / setModel() ─────────────────────────────────────────────────────

namespace {

// CurveModel with a model-held offset, so that setModel() changes the results.
struct OffsetCurveModel : CurveModel {
  OffsetCurveModel() = default;
  explicit OffsetCurveModel(double offset) : offset_(offset) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    CurveModel::residual(x, in, obs, res);
    res[0] += offset_;
  }
  double offset_ = 0.0;
};

template <class Cost>
System evaluate(Cost& cost, const double* x) {
  System system;
  cost.computeLinearSystem(x, system.jtj.data(), system.jtb.data(), system.cost);
  return system;
}

}  // namespace

TEST(ReusableCost, SetDataEqualsAFreshCostWhetherTheDataGrowsOrShrinks) {
  const Data big = makeData(30000, kTruth, 31);
  const Data small = makeData(700, kTruth, 37);
  const auto context = makeContext(0);
  using Analytic = AnalyticalCost<OffsetCurveModel, double>;
  using Central = NumericalCostCentral<OffsetCurveModel, double>;

  Analytic analytic(small.input.data(), small.observations.data(), 700, 2, 2, kParams, OffsetCurveModel{0.1}, 0,
                    context);
  Central central(small.input.data(), small.observations.data(), 700, 2, 2, kParams, OffsetCurveModel{0.1}, 0, context);
  central.setLossFunction(std::make_shared<CauchyLoss<double>>(0.3));
  evaluate(analytic, kX);  // allocates for 700 elements
  evaluate(central, kX);

  // 700 -> 30000 (more partitions, same chunk size) -> 3 (smaller chunks) -> 0 -> 700.
  const std::vector<std::pair<const Data*, size_t>> steps = {{&big, 30000}, {&small, 3}, {&small, 0}, {&small, 700}};
  for (const auto& [data, n] : steps) {
    SCOPED_TRACE(testing::Message() << "n=" << n);
    analytic.setData(data->input.data(), data->observations.data(), n);
    central.setData(data->input.data(), data->observations.data(), n);
    EXPECT_EQ(analytic.numElements(), n);

    Analytic fresh_analytic(data->input.data(), data->observations.data(), n, 2, 2, kParams, OffsetCurveModel{0.1}, 0,
                            context);
    Central fresh_central(data->input.data(), data->observations.data(), n, 2, 2, kParams, OffsetCurveModel{0.1}, 0,
                          context);
    fresh_central.setLossFunction(std::make_shared<CauchyLoss<double>>(0.3));

    const System a = evaluate(analytic, kX), fa = evaluate(fresh_analytic, kX);
    const System c = evaluate(central, kX), fc = evaluate(fresh_central, kX);
    EXPECT_TRUE(sameBits(a.jtj, fa.jtj));
    EXPECT_TRUE(sameBits(a.jtb, fa.jtb));
    EXPECT_TRUE(sameBits(a.cost, fa.cost));
    EXPECT_TRUE(sameBits(c.jtj, fc.jtj));
    EXPECT_TRUE(sameBits(c.cost, fc.cost));
    EXPECT_TRUE(sameBits(analytic.computeCost(kX), fresh_analytic.computeCost(kX)));
  }
}

TEST(ReusableCost, SetDataDropsTheNoiseModel) {
  const Data data = makeData(400, kTruth, 41);
  const auto context = makeContext(0);
  Curve weighted(data.input.data(), data.observations.data(), 400, 2, 2, kParams, CurveModel{}, 0, context);
  weighted.setWeights(std::vector<double>(400, 4.0));
  weighted.setData(data.input.data(), data.observations.data(), 400);
  Curve plain(data.input.data(), data.observations.data(), 400, 2, 2, kParams, CurveModel{}, 0, context);
  EXPECT_TRUE(sameBits(weighted.computeCost(kX), plain.computeCost(kX)));
}

TEST(ReusableCost, SetModelEqualsAFreshCost) {
  const Data data = makeData(2500, kTruth, 43);
  const auto context = makeContext(0);
  using Central = NumericalCostCentral<OffsetCurveModel, double>;
  Central reused(data.input.data(), data.observations.data(), 2500, 2, 2, kParams, OffsetCurveModel{0.0}, 0, context);
  // Before the first evaluation (the model is not built yet) and after it.
  reused.setModel(OffsetCurveModel{0.2});
  const System first = evaluate(reused, kX);
  reused.setModel(OffsetCurveModel{-0.3});
  const System second = evaluate(reused, kX);

  Central fresh_first(data.input.data(), data.observations.data(), 2500, 2, 2, kParams, OffsetCurveModel{0.2}, 0,
                      context);
  Central fresh_second(data.input.data(), data.observations.data(), 2500, 2, 2, kParams, OffsetCurveModel{-0.3}, 0,
                       context);
  EXPECT_TRUE(sameBits(first.jtj, evaluate(fresh_first, kX).jtj));
  EXPECT_TRUE(sameBits(first.cost, evaluate(fresh_first, kX).cost));
  EXPECT_TRUE(sameBits(second.jtj, evaluate(fresh_second, kX).jtj));
  EXPECT_TRUE(sameBits(second.cost, evaluate(fresh_second, kX).cost));
}

TEST(ReusableCost, AFusedCostSeesItsCostsNewData) {
  const Data a = makeData(3000, kTruth, 47);
  const Data b = makeData(1800, kTruth, 53);
  const auto context = makeContext(0);
  Problem problem(a, 3000, context);
  const auto fused = makeFusedCost<double>(problem.curve, problem.central, problem.prior);
  System before;
  fused->computeLinearSystem(kX, before.jtj.data(), before.jtb.data(), before.cost);

  problem.curve->setData(b.input.data(), b.observations.data(), 1800);
  const System expected = sumOfParts({problem.curve.get(), problem.central.get(), problem.prior.get()}, kX);
  System got;
  fused->computeLinearSystem(kX, got.jtj.data(), got.jtb.data(), got.cost);
  EXPECT_TRUE(sameBits(got.jtj, expected.jtj));
  EXPECT_TRUE(sameBits(got.cost, expected.cost));
  EXPECT_FALSE(sameBits(got.cost, before.cost));
}
