// The asynchronous cost interface (submit*/wait), the optimizers' use of it, and speculative linear
// systems in Levenberg-Marquardt. Every comparison between the two ways of running is bitwise.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "manifold_models.hh"
#include "moptim/AnalyticalCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/GaussNewton.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/Observer.hh"

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
  return std::make_shared<ExecutionContext>(
      sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()}, threads);
}

template <class T>
bool sameBits(const T& a, const T& b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool sameBits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

struct System {
  std::vector<double> jtj = std::vector<double>(kParams * kParams, 7.0);
  std::vector<double> jtb = std::vector<double>(kParams, 7.0);
  double cost = 7.0;
};

const double kX[kParams] = {0.3, -0.7, 0.2, 1.1};

}  // namespace

// ─── submit / wait ───────────────────────────────────────────────────────────────────────────────

TEST(AsyncCost, SubmitAndWaitEqualCompute) {
  const double truth[kParams] = {0.5, -0.4, 0.1, 0.9};
  const Data data = makeData(5000, truth, 3);
  const auto shared = makeContext(0);

  for (const bool separate_contexts : {false, true}) {
    SCOPED_TRACE(testing::Message() << "separate_contexts=" << separate_contexts);
    const auto context = [&] { return separate_contexts ? makeContext(0) : shared; };
    AnalyticalCost<CurveModel, double> analytical(data.input.data(), data.observations.data(), 5000, 2, 2, kParams,
                                                  CurveModel{}, 0, context());
    NumericalCostCentral<CurveModel, double> central(data.input.data(), data.observations.data(), 5000, 2, 2, kParams,
                                                     CurveModel{}, 0, context());
    NumericalCostForwardEuler<CurveModel, double> forward(data.input.data(), data.observations.data(), 5000, 2, 2,
                                                          kParams, CurveModel{}, 0, context());
    forward.setWeights(std::vector<double>(5000, 1.5));
    forward.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
    ICost<double>* costs[] = {&analytical, &central, &forward};

    System expected[3];
    double expected_cost[3];
    for (int i = 0; i < 3; ++i) {
      costs[i]->computeLinearSystem(kX, expected[i].jtj.data(), expected[i].jtb.data(), expected[i].cost);
      expected_cost[i] = costs[i]->computeCost(kX);
    }

    // All the systems in flight at once, then all the costs.
    System got[3];
    for (int i = 0; i < 3; ++i) {
      costs[i]->submitLinearSystem(kX, got[i].jtj.data(), got[i].jtb.data(), &got[i].cost);
    }
    for (int i = 0; i < 3; ++i) costs[i]->wait();
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(sameBits(got[i].jtj, expected[i].jtj)) << i;
      EXPECT_TRUE(sameBits(got[i].jtb, expected[i].jtb)) << i;
      EXPECT_TRUE(sameBits(got[i].cost, expected[i].cost)) << i;
    }

    double got_cost[3] = {-1, -1, -1};
    for (int i = 0; i < 3; ++i) costs[i]->submitCost(kX, &got_cost[i]);
    for (int i = 0; i < 3; ++i) costs[i]->wait();
    for (int i = 0; i < 3; ++i) EXPECT_TRUE(sameBits(got_cost[i], expected_cost[i])) << i;

    for (int i = 0; i < 3; ++i) costs[i]->wait();  // nothing pending: harmless
  }
}

TEST(AsyncCost, SubmitCountsOneLaunchAndXMayBeReused) {
  const double truth[kParams] = {0.5, -0.4, 0.1, 0.9};
  const Data data = makeData(2000, truth, 4);
  const auto context = makeContext(0);
  AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), 2000, 2, 2, kParams,
                                          CurveModel{}, 0, context);
  System reference;
  cost.computeLinearSystem(kX, reference.jtj.data(), reference.jtb.data(), reference.cost);

  double x[kParams] = {kX[0], kX[1], kX[2], kX[3]};
  System system;
  const uint64_t before = context->launches();
  cost.submitLinearSystem(x, system.jtj.data(), system.jtb.data(), &system.cost);
  EXPECT_EQ(context->launches(), before + 1);
  for (double& entry : x) entry = 99.0;  // x is read at submit
  cost.wait();
  EXPECT_EQ(context->launches(), before + 1);
  EXPECT_TRUE(sameBits(system.jtj, reference.jtj));
  EXPECT_TRUE(sameBits(system.cost, reference.cost));
}

// Calling anything on a cost with an evaluation outstanding completes that one first.
TEST(AsyncCost, ANewCallCompletesThePendingOne) {
  const double truth[kParams] = {0.5, -0.4, 0.1, 0.9};
  const Data data = makeData(2000, truth, 5);
  AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), 2000, 2, 2, kParams,
                                          CurveModel{}, 0, makeContext(0));
  const double other[kParams] = {0.1, 0.2, 0.3, 0.4};
  System reference_x, reference_other;
  cost.computeLinearSystem(kX, reference_x.jtj.data(), reference_x.jtb.data(), reference_x.cost);
  cost.computeLinearSystem(other, reference_other.jtj.data(), reference_other.jtb.data(), reference_other.cost);

  System first, second;
  cost.submitLinearSystem(kX, first.jtj.data(), first.jtb.data(), &first.cost);
  cost.submitLinearSystem(other, second.jtj.data(), second.jtb.data(), &second.cost);  // completes `first`
  EXPECT_TRUE(sameBits(first.jtj, reference_x.jtj));
  EXPECT_TRUE(sameBits(first.cost, reference_x.cost));
  cost.wait();
  EXPECT_TRUE(sameBits(second.jtj, reference_other.jtj));

  double pending_cost = -1.0;
  cost.submitCost(kX, &pending_cost);
  const double direct = cost.computeCost(other);  // completes the submitted cost, then evaluates
  EXPECT_TRUE(sameBits(pending_cost, reference_x.cost));
  EXPECT_TRUE(sameBits(direct, reference_other.cost));
}

TEST(AsyncCost, NoElementsAreWrittenAtSubmit) {
  const auto context = makeContext(0);
  AnalyticalCost<CurveModel, double> cost(nullptr, nullptr, 0, 2, 2, kParams, CurveModel{}, 0, context);
  System system;
  cost.submitLinearSystem(kX, system.jtj.data(), system.jtb.data(), &system.cost);
  EXPECT_EQ(system.cost, 0.0);
  EXPECT_EQ(system.jtb[0], 0.0);
  double value = 5.0;
  cost.submitCost(kX, &value);
  EXPECT_EQ(value, 0.0);
  cost.wait();
  EXPECT_EQ(context->launches(), 0u);
}

TEST(AsyncCost, DestroyingACostWithAPendingEvaluationIsSafe) {
  const double truth[kParams] = {0.5, -0.4, 0.1, 0.9};
  const Data data = makeData(20000, truth, 6);
  System system;
  {
    AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), 20000, 2, 2, kParams,
                                            CurveModel{}, 0, makeContext(0));
    cost.submitLinearSystem(kX, system.jtj.data(), system.jtb.data(), &system.cost);
  }
  SUCCEED();
}

// ─── The optimizers over several costs ───────────────────────────────────────────────────────────

namespace {

/// Hides the asynchronous overrides of a cost: submit/wait fall back to the synchronous defaults.
class SyncOnly : public ICost<double> {
 public:
  explicit SyncOnly(std::shared_ptr<ICost<double>> inner)
      : ICost<double>(inner->inputDim(), inner->observationDim(), inner->paramDim(), inner->numElements(),
                      inner->activeParamDim()),
        inner_(std::move(inner)) {}
  double computeCost(const double* x) override { return inner_->computeCost(x); }
  void computeLinearSystem(const double* x, double* jtj, double* jtb, double& cost) override {
    inner_->computeLinearSystem(x, jtj, jtb, cost);
  }

 private:
  std::shared_ptr<ICost<double>> inner_;
};

struct OptRun {
  std::vector<double> x;
  Status status;
  size_t iterations;
  double final_cost;
};

bool sameRun(const OptRun& a, const OptRun& b) {
  return sameBits(a.x, b.x) && a.status == b.status && a.iterations == b.iterations && sameBits(a.final_cost, b.final_cost);
}

struct Problem {
  Data first, second;
  std::vector<double> start{1.5, 1.0, -1.0, 2.0};
};

Problem makeProblem() {
  const double truth[kParams] = {0.5, -0.4, 0.1, 0.9};
  return {makeData(3000, truth, 8), makeData(900, truth, 9)};
}

std::vector<std::shared_ptr<ICost<double>>> makeCosts(const Problem& p, const std::shared_ptr<ExecutionContext>& ctx,
                                                      bool sync_only) {
  auto a = std::make_shared<AnalyticalCost<CurveModel, double>>(p.first.input.data(), p.first.observations.data(), 3000,
                                                                2, 2, kParams, CurveModel{}, 0, ctx);
  auto b = std::make_shared<NumericalCostCentral<CurveModel, double>>(
      p.second.input.data(), p.second.observations.data(), 900, 2, 2, kParams, CurveModel{}, 0, ctx);
  b->setWeights(std::vector<double>(900, 0.5));
  std::vector<std::shared_ptr<ICost<double>>> costs{a, b};
  if (sync_only) {
    for (auto& cost : costs) cost = std::make_shared<SyncOnly>(cost);
  }
  return costs;
}

template <class Optimizer>
OptRun runOptimizer(Optimizer& optimizer, std::vector<double> x) {
  const Result<double> result = optimizer.optimize(x.data());
  return {x, result.status, result.iterations, result.final_cost};
}

}  // namespace

TEST(AsyncOptimizer, SeveralCostsGiveTheSameBitsAsTheSynchronousPath) {
  const Problem problem = makeProblem();
  const auto context = makeContext(0);
  for (const bool gauss_newton : {false, true}) {
    SCOPED_TRACE(testing::Message() << "gauss_newton=" << gauss_newton);
    OptRun runs[2];
    for (const bool sync_only : {false, true}) {
      const auto costs = makeCosts(problem, context, sync_only);
      if (gauss_newton) {
        GaussNewton<double> solver(kParams);
        for (const auto& cost : costs) solver.addCost(cost);
        solver.setMaxIterations(8);
        runs[sync_only] = runOptimizer(solver, problem.start);
      } else {
        LevenbergMarquardt<double> solver(kParams);
        for (const auto& cost : costs) solver.addCost(cost);
        solver.setMaxIterations(20);
        runs[sync_only] = runOptimizer(solver, problem.start);
      }
    }
    EXPECT_TRUE(sameRun(runs[0], runs[1]));
  }
}

// ─── Speculative linear systems ──────────────────────────────────────────────────────────────────

namespace {

class Recorder : public IOptimizerObserver<double> {
 public:
  void onIteration(const IterationEvent<double>& event) override {
    if (event.phase == Phase::REJECTED) ++rejected;
    if (event.phase == Phase::ACCEPTED) ++accepted;
  }
  void onLinearSystem(const LinearSystemEvent<double>& event) override { systems.push_back(event.cost); }
  size_t rejected = 0, accepted = 0;
  std::vector<double> systems;
};

struct SpeculationOutcome {
  OptRun run;
  uint64_t launches;
  Recorder recorder;
};

}  // namespace

TEST(SpeculativeLM, SameBitsAndFewerLaunches) {
  const Problem problem = makeProblem();
  const auto context = makeContext(0);
  SpeculationOutcome outcome[2];
  for (const bool speculative : {false, true}) {
    const auto costs = makeCosts(problem, context, false);
    LevenbergMarquardt<double> solver(kParams);
    for (const auto& cost : costs) solver.addCost(cost);
    solver.setMaxIterations(20);
    solver.setSpeculativeLinearSystems(speculative);
    EXPECT_EQ(solver.speculativeLinearSystems(), speculative);
    solver.setObserver(&outcome[speculative].recorder);
    const uint64_t before = context->launches();
    outcome[speculative].run = runOptimizer(solver, problem.start);
    outcome[speculative].launches = context->launches() - before;
  }
  EXPECT_TRUE(sameRun(outcome[0].run, outcome[1].run));
  EXPECT_GT(outcome[0].run.iterations, 3u);
  EXPECT_LT(outcome[1].launches, outcome[0].launches)
      << "off " << outcome[0].launches << " on " << outcome[1].launches;
  // One linear system per iteration that uses one, either way, with the same cost.
  EXPECT_EQ(outcome[0].recorder.systems.size(), outcome[1].recorder.systems.size());
  for (size_t i = 0; i < std::min(outcome[0].recorder.systems.size(), outcome[1].recorder.systems.size()); ++i) {
    EXPECT_TRUE(sameBits(outcome[0].recorder.systems[i], outcome[1].recorder.systems[i])) << i;
  }
  EXPECT_EQ(outcome[0].recorder.accepted, outcome[1].recorder.accepted);
  EXPECT_EQ(outcome[0].recorder.rejected, outcome[1].recorder.rejected);
}

namespace {

// y = x1 * exp(x0 * t): a start far from the solution overshoots, so some trials are rejected.
struct ExpModel {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - x[1] * std::exp(x[0] * in[0]);
  }
  void jacobian(const double* x, const double* in, const double* /*obs*/, double* jac) const {
    const double e = std::exp(x[0] * in[0]);
    jac[0] = -x[1] * in[0] * e;
    jac[1] = -e;
  }
};

}  // namespace

// Trials that are rejected: the speculative system of a rejected trial is dropped, and the results
// are still those of the plain path.
TEST(SpeculativeLM, RejectedTrialsGiveTheSameBits) {
  std::vector<double> t, y;
  for (int i = 0; i < 300; ++i) {
    t.push_back(0.01 * i);
    y.push_back(0.8 * std::exp(1.2 * t.back()));
  }
  const auto context = makeContext(0);
  size_t total_rejected = 0;
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> offset(-4.0, 4.0);
  for (int trial = 0; trial < 20; ++trial) {
    const std::vector<double> start{offset(rng), offset(rng)};
    OptRun runs[2];
    Recorder recorders[2];
    uint64_t launches[2];
    for (const bool speculative : {false, true}) {
      LevenbergMarquardt<double> solver(2);
      solver.addCost(std::make_shared<AnalyticalCost<ExpModel, double>>(t.data(), y.data(), 300, 1, 1, 2,
                                                                         ExpModel{}, 0, context));
      solver.setMaxIterations(25);
      solver.setSpeculativeLinearSystems(speculative);
      solver.setObserver(&recorders[speculative]);
      const uint64_t before = context->launches();
      runs[speculative] = runOptimizer(solver, start);
      launches[speculative] = context->launches() - before;
    }
    EXPECT_TRUE(sameRun(runs[0], runs[1])) << "start " << trial;
    EXPECT_EQ(recorders[0].systems.size(), recorders[1].systems.size());
    EXPECT_EQ(recorders[0].rejected, recorders[1].rejected);
    total_rejected += recorders[0].rejected;
  }
  EXPECT_GT(total_rejected, 0u) << "the starting points never rejected a trial";
}

namespace {
// Pulls the translation of an SE(3) pose (entries 9..11 of x) to a measured value. A kernel cannot be
// named after a function-local type, so it lives here.
struct Prior {
  void setState(const double*) {}
  void residual(const double* x, const double*, const double* obs, double* res) const {
    for (int i = 0; i < 3; ++i) res[i] = obs[i] - x[9 + i];
  }
};
}  // namespace

// A pose on a manifold with an analytical cloud and a numerical prior, as mslam runs it.
TEST(SpeculativeLM, ManifoldWithTwoCosts) {
  using namespace manifold_models;
  const std::shared_ptr<const IManifold<double>> manifold = std::make_shared<SE3Manifold>();
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> coordinate(-50.0, 50.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  constexpr size_t kN = 1200;
  std::vector<double> inputs(6 * kN), offsets(kN);
  Eigen::Matrix<double, 6, 1> truth_tangent;
  truth_tangent << 0.1, -0.2, 0.3, 0.05, -0.03, 0.08;
  const Eigen::Matrix<double, 12, 1> truth = poseFromTangent(truth_tangent);
  for (size_t i = 0; i < kN; ++i) {
    const Eigen::Vector3d p(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d n = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
    Eigen::Map<Eigen::Vector3d>{&inputs[6 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&inputs[6 * i + 3]} = n;
    offsets[i] = n.dot(Eigen::Map<const Eigen::Matrix3d>(truth.data()) * p +
                       Eigen::Map<const Eigen::Vector3d>(truth.data() + 9)) +
                 0.01 * gauss(rng);
  }
  const double prior[3] = {0.1, -0.2, 0.3};
  const double unused_input = 0.0;
  Eigen::Matrix<double, 6, 1> start_tangent;
  start_tangent << 0.02, -0.03, 0.04, 0.01, -0.02, 0.015;

  const auto context = makeContext(0);
  OptRun runs[2];
  uint64_t launches[2];
  for (const bool speculative : {false, true}) {
    LevenbergMarquardt<double> solver(manifold);
    solver.addCost(std::make_shared<AnalyticalCost<PointToPlane, double>>(inputs.data(), offsets.data(), kN, 6, 1,
                                                                          manifold, PointToPlane{}, 0, context));
    solver.addCost(std::make_shared<NumericalCostCentral<Prior, double>>(&unused_input, prior, 1, 1, 3, manifold,
                                                                         Prior{}, 0, context));
    solver.setMaxIterations(5);
    solver.setSpeculativeLinearSystems(speculative);
    const Eigen::Matrix<double, 12, 1> x0 = poseFromTangent(start_tangent);
    const uint64_t before = context->launches();
    runs[speculative] = runOptimizer(solver, std::vector<double>(x0.data(), x0.data() + 12));
    launches[speculative] = context->launches() - before;
  }
  EXPECT_TRUE(sameRun(runs[0], runs[1]));
  EXPECT_LT(launches[1], launches[0]) << launches[0] << " vs " << launches[1];
}

namespace {

// A custom cost that is not a ChunkedCost: f(x) = (x - 3)^2, one parameter.
class QuadraticCost : public ICost<double> {
 public:
  QuadraticCost() : ICost<double>(1, 1, 1, 1) {}
  double computeCost(const double* x) override {
    ++cost_calls;
    return (x[0] - 3.0) * (x[0] - 3.0);
  }
  void computeLinearSystem(const double* x, double* jtj, double* jtb, double& cost) override {
    ++system_calls;
    jtj[0] = 1.0;
    jtb[0] = x[0] - 3.0;
    cost = (x[0] - 3.0) * (x[0] - 3.0);
  }
  int cost_calls = 0;
  int system_calls = 0;
};

}  // namespace

TEST(SpeculativeLM, ACustomCostWorksWithSpeculationOffAndOn) {
  OptRun runs[2];
  int cost_calls[2], system_calls[2];
  for (const bool speculative : {false, true}) {
    const auto cost = std::make_shared<QuadraticCost>();
    LevenbergMarquardt<double> solver(1);
    solver.addCost(cost);
    solver.setSpeculativeLinearSystems(speculative);
    runs[speculative] = runOptimizer(solver, {10.0});
    cost_calls[speculative] = cost->cost_calls;
    system_calls[speculative] = cost->system_calls;
  }
  EXPECT_TRUE(sameRun(runs[0], runs[1]));
  EXPECT_NEAR(runs[1].x[0], 3.0, 1e-3);
  // Off: a cost per trial. On: a linear system per trial (the cost is then taken from it).
  EXPECT_GT(cost_calls[0], 0);
  EXPECT_EQ(cost_calls[1], 0);
  EXPECT_GE(system_calls[1], system_calls[0]);
}
