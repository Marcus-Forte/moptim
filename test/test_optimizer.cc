#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/EigenSolver.hh"
#include "moptim/GaussNewton.hh"
#include "moptim/ICost.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/Observer.hh"
#include "test_models.hh"

using namespace moptim;
using namespace test_models;

namespace {

class RecordingObserver : public IOptimizerObserver<double> {
 public:
  void onIteration(const IterationEvent<double>& event) override { iterations.push_back(event); }
  void onLinearSystem(const LinearSystemEvent<double>& event) override { linear_systems.push_back(event); }

  std::vector<IterationEvent<double>> iterations;
  std::vector<LinearSystemEvent<double>> linear_systems;
};

Eigen::Matrix3d positiveDefiniteMatrix() {
  Eigen::Matrix3d matrix;
  matrix << 4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0;
  return matrix;
}

// y = (x[0] + x[1]) * input: the two parameters only enter through their sum, so the two columns
// of the jacobian are equal and J^T J is singular, although its diagonal is positive.
struct SumOfSlopesModel {
  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* input, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] + x[1]) * input[0];
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    jac[0] = -input[0];
    jac[1] = -input[0];
  }
};

// The squared inputs add up to 9, a perfect square: the Cholesky factorization of J^T J then
// meets an exactly zero pivot instead of a rounding-sized one.
const double kSlopeInput[] = {1.0, 2.0, 2.0};
const double kSlopeObservations[] = {3.0, 6.5, 5.5};

std::shared_ptr<AnalyticalCost<SumOfSlopesModel, double>> makeSumOfSlopesCost() {
  return std::make_shared<AnalyticalCost<SumOfSlopesModel, double>>(kSlopeInput, kSlopeObservations, 3, 1, 1, 2);
}

std::shared_ptr<AnalyticalCost<SimpleModel<double>, double>> makeSimpleCost() {
  return std::make_shared<AnalyticalCost<SimpleModel<double>, double>>(
      TestData<double>::x_data_, TestData<double>::y_data_, TestData<double>::num_measurements, 1, 1, 2);
}

// Reports the same one-parameter linear system wherever it is asked, and a cost that is worse than
// that system's at every trial point. It counts how often it is evaluated.
class StubbornCost : public ICost<double> {
 public:
  StubbornCost() : ICost<double>(1, 1, 1, 1) {}

  double computeCost(const double* /*x*/) override {
    ++cost_calls;
    return 2.0;
  }

  void computeLinearSystem(const double* /*x*/, double* JTJ, double* JTb, double& cost) override {
    ++linear_system_calls;
    JTJ[0] = 1.0;
    JTb[0] = -1.0;
    cost = 1.0;
  }

  int cost_calls = 0;
  int linear_system_calls = 0;
};

}  // namespace

TEST(EigenSolver, SolvesForTheStep) {
  const Eigen::Matrix3d A = positiveDefiniteMatrix();
  const Eigen::Vector3d b(1.0, -2.0, 0.5);
  Eigen::Vector3d x;

  EigenSolver<double> solver(3);
  ASSERT_TRUE(solver.solve(A.data(), b.data(), x.data()));

  // A * x = -b
  EXPECT_LE((A * x + b).norm(), 1e-12);
}

TEST(EigenSolver, ReadsOnlyTheLowerTriangle) {
  const Eigen::Matrix3d A = positiveDefiniteMatrix();
  Eigen::Matrix3d lower_only = A;
  lower_only.triangularView<Eigen::StrictlyUpper>().setConstant(1e6);
  const Eigen::Vector3d b(1.0, -2.0, 0.5);

  EigenSolver<double> solver(3);
  Eigen::Vector3d expected;
  Eigen::Vector3d x;
  ASSERT_TRUE(solver.solve(A.data(), b.data(), expected.data()));
  ASSERT_TRUE(solver.solve(lower_only.data(), b.data(), x.data()));

  EXPECT_EQ(x, expected);
}

TEST(EigenSolver, ReportsMatricesThatAreNotPositiveDefinite) {
  const Eigen::Vector2d b(1.0, 1.0);
  Eigen::Vector2d x;
  EigenSolver<double> solver(2);

  Eigen::Matrix2d indefinite;
  indefinite << 1.0, 0.0, 0.0, -1.0;
  EXPECT_FALSE(solver.solve(indefinite.data(), b.data(), x.data()));

  Eigen::Matrix2d singular;
  singular << 1.0, 0.0, 0.0, 0.0;
  EXPECT_FALSE(solver.solve(singular.data(), b.data(), x.data()));

  // A failure leaves the solver usable.
  const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
  ASSERT_TRUE(solver.solve(identity.data(), b.data(), x.data()));
  EXPECT_EQ(x, Eigen::Vector2d(-1.0, -1.0));
}

TEST(EigenSolver, ReportsNonFiniteSystems) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
  Eigen::Vector2d x;
  EigenSolver<double> solver(2);

  const Eigen::Vector2d nan_b(1.0, nan);
  EXPECT_FALSE(solver.solve(identity.data(), nan_b.data(), x.data()));

  Eigen::Matrix2d nan_A = identity;
  nan_A(1, 1) = nan;
  const Eigen::Vector2d b(1.0, 1.0);
  EXPECT_FALSE(solver.solve(nan_A.data(), b.data(), x.data()));
}

TEST(GaussNewton, StopsWithNumericErrorWhenTheSystemCannotBeSolved) {
  const auto cost = makeSumOfSlopesCost();
  GaussNewton<double> solver(2);
  solver.addCost(cost);
  RecordingObserver observer;
  solver.setObserver(&observer);

  const Eigen::Vector2d start(0.5, -3.0);
  Eigen::Vector2d x = start;
  const auto result = solver.optimize(x.data());

  EXPECT_EQ(result.status, Status::NUMERIC_ERROR);
  EXPECT_EQ(result.iterations, 1u);
  // No step was taken.
  EXPECT_EQ(x, start);
  EXPECT_DOUBLE_EQ(result.final_cost, cost->computeCost(x.data()));

  ASSERT_EQ(observer.iterations.size(), 2u);
  EXPECT_EQ(observer.iterations[0].phase, Phase::REJECTED);
  EXPECT_EQ(observer.iterations[0].status, Status::NUMERIC_ERROR);
  EXPECT_EQ(observer.iterations[1].phase, Phase::FINISHED);
  EXPECT_EQ(observer.iterations[1].status, Status::NUMERIC_ERROR);
}

// Damping makes the same system positive definite, so Levenberg-Marquardt gets through it.
TEST(LevenbergMarquardt, SolvesTheRankDeficientProblemGaussNewtonCannot) {
  const auto cost = makeSumOfSlopesCost();
  LevenbergMarquardt<double> solver(2);
  solver.addCost(cost);

  Eigen::Vector2d x(0.5, -3.0);
  const auto result = solver.optimize(x.data());

  // Least-squares slope: sum(input * obs) / sum(input^2).
  const double slope = (1.0 * 3.0 + 2.0 * 6.5 + 2.0 * 5.5) / 9.0;
  EXPECT_NE(result.status, Status::NUMERIC_ERROR);
  EXPECT_NEAR(x[0] + x[1], slope, 1e-6);
  EXPECT_DOUBLE_EQ(result.final_cost, cost->computeCost(x.data()));
}

TEST(Optimizer, FinalCostIsTheCostAtTheReturnedPoint) {
  {
    const auto cost = makeSimpleCost();
    GaussNewton<double> solver(2);
    solver.addCost(cost);
    Eigen::Vector2d x(0.9, 0.2);
    const auto result = solver.optimize(x.data());
    EXPECT_DOUBLE_EQ(result.final_cost, cost->computeCost(x.data()));
  }
  {
    const auto cost = makeSimpleCost();
    LevenbergMarquardt<double> solver(2);
    solver.addCost(cost);
    Eigen::Vector2d x(0.9, 0.2);
    const auto result = solver.optimize(x.data());
    EXPECT_DOUBLE_EQ(result.final_cost, cost->computeCost(x.data()));
  }
  {
    // Out of iterations right after an accepted step.
    const auto cost = makeSimpleCost();
    LevenbergMarquardt<double> solver(2);
    solver.addCost(cost);
    solver.setMaxIterations(2);
    Eigen::Vector2d x(0.9, 0.2);
    const auto result = solver.optimize(x.data());
    EXPECT_EQ(result.status, Status::MAX_ITERATIONS_REACHED);
    EXPECT_DOUBLE_EQ(result.final_cost, cost->computeCost(x.data()));
  }
}

// When every trial of an iteration is rejected, x does not move, so the next iteration of the same
// run reuses the linear system and only tries larger damping factors.
TEST(LevenbergMarquardt, AssemblesTheLinearSystemOnceWhileEveryTrialIsRejected) {
  const auto cost = std::make_shared<StubbornCost>();
  LevenbergMarquardt<double> solver(1);
  solver.addCost(cost);
  solver.setMaxIterations(4);
  RecordingObserver observer;
  solver.setObserver(&observer);

  double x = 0.25;
  const auto result = solver.optimize(&x);

  EXPECT_EQ(result.status, Status::MAX_ITERATIONS_REACHED);
  EXPECT_EQ(result.iterations, 4u);
  EXPECT_EQ(x, 0.25);
  EXPECT_EQ(cost->linear_system_calls, 1);
  EXPECT_EQ(observer.linear_systems.size(), 1u);
  // Three trials per iteration, and no further evaluation to report the final cost: with x where
  // it started, that is the cost of the linear system.
  EXPECT_EQ(cost->cost_calls, 4 * 3);
  EXPECT_EQ(result.final_cost, 1.0);

  // 12 rejected trials with ever larger damping, then the FINISHED event of the run.
  ASSERT_EQ(observer.iterations.size(), 13u);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(observer.iterations[i].phase, Phase::REJECTED);
    if (i > 0) {
      EXPECT_GT(observer.iterations[i].lambda, observer.iterations[i - 1].lambda);
    }
  }
  EXPECT_EQ(observer.iterations.back().phase, Phase::FINISHED);
}

// A manual step() and the first iteration of a run cannot assume anything about what the caller
// did in between (new data behind the same pointers, a different x): they always reassemble.
TEST(LevenbergMarquardt, StepsAndNewRunsAlwaysAssembleTheLinearSystem) {
  const auto cost = std::make_shared<StubbornCost>();
  LevenbergMarquardt<double> solver(1);
  solver.addCost(cost);
  solver.setMaxIterations(2);

  double x = 0.25;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(solver.step(&x), Status::STEP_OK);
  }
  EXPECT_EQ(cost->linear_system_calls, 3);

  solver.optimize(&x);
  solver.optimize(&x);
  EXPECT_EQ(cost->linear_system_calls, 5);
}
