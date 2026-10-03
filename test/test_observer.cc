#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "AnalyticalCost.hh"
#include "LevenbergMarquardt.hh"
#include "Observer.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

namespace {

template <class T>
class RecordingObserver : public IOptimizerObserver<T> {
 public:
  void onIteration(const IterationEvent<T>& event) override { iterations.push_back(event); }
  void onLinearSystem(const LinearSystemEvent<T>& event) override { linear_systems.push_back(event); }

  std::vector<IterationEvent<T>> iterations;
  std::vector<LinearSystemEvent<T>> linear_systems;
};

/**
 * @brief Minimal console logger written directly against IOptimizerObserver.
 *
 * This is the pattern to use for console output now that moptim ships no
 * logging framework: implement the observer and format the events yourself.
 * Taking an ostream (defaulting to std::cout) keeps it usable and testable.
 */
template <class T>
class ConsoleLoggingObserver : public IOptimizerObserver<T> {
 public:
  explicit ConsoleLoggingObserver(std::ostream& out = std::cout) : out_(out) {}

  void onLinearSystem(const LinearSystemEvent<T>& event) override {
    out_ << "[linear system] iter " << event.iteration << " cost " << event.cost << " ("
         << event.elapsed.count() / 1000 << " us)" << std::endl;
  }

  void onIteration(const IterationEvent<T>& event) override {
    out_ << "[iteration] iter " << event.iteration << " phase " << toString(event.phase) << " status "
         << static_cast<int>(event.status) << " cost " << event.cost << " rho " << event.rho << " lambda "
         << event.lambda << " delta " << event.delta_norm << std::endl;
  }

 private:
  static const char* toString(Phase phase) {
    switch (phase) {
      case Phase::LINEAR_SYSTEM:
        return "LINEAR_SYSTEM";
      case Phase::TRIAL:
        return "TRIAL";
      case Phase::ACCEPTED:
        return "ACCEPTED";
      case Phase::REJECTED:
        return "REJECTED";
      case Phase::FINISHED:
        return "FINISHED";
    }
    return "UNKNOWN";
  }

  std::ostream& out_;
};

std::shared_ptr<AnalyticalCost<SimpleModel<double>, double>> makeSimpleCost() {
  return std::make_shared<AnalyticalCost<SimpleModel<double>, double>>(
      TestData<double>::x_data_, TestData<double>::y_data_, TestData<double>::num_measurements, 1, 1, 2,
      SimpleModel<double>{});
}

}  // namespace

TEST(TestObserver, LevenbergMarquardtEmitsStructuredEvents) {
  Eigen::VectorXd x{{0.9, 0.2}};

  LevenbergMarquardt<double> solver(2);
  solver.addCost(makeSimpleCost());

  RecordingObserver<double> observer;
  solver.setObserver(&observer);

  const auto result = solver.optimize(x.data());

  ASSERT_FALSE(observer.iterations.empty());
  ASSERT_FALSE(observer.linear_systems.empty());

  // A FINISHED event is always emitted, exactly once, and mirrors the status.
  const auto finished = std::count_if(observer.iterations.begin(), observer.iterations.end(),
                                      [](const IterationEvent<double>& e) { return e.phase == Phase::FINISHED; });
  EXPECT_EQ(finished, 1);
  EXPECT_EQ(observer.iterations.back().phase, Phase::FINISHED);
  EXPECT_EQ(observer.iterations.back().status, result.status);
  EXPECT_GT(result.iterations, 0u);
  EXPECT_GT(result.final_cost, 0.0);
}

TEST(TestObserver, EventsCarryFinitePayload) {
  Eigen::VectorXd x{{0.9, 0.2}};

  LevenbergMarquardt<double> solver(2);
  solver.addCost(makeSimpleCost());

  RecordingObserver<double> observer;
  solver.setObserver(&observer);
  solver.optimize(x.data());

  for (const auto& event : observer.iterations) {
    EXPECT_TRUE(std::isfinite(event.cost));
    EXPECT_GE(event.elapsed.count(), 0);
  }
  for (const auto& event : observer.linear_systems) {
    EXPECT_TRUE(std::isfinite(event.cost));
    EXPECT_GE(event.elapsed.count(), 0);
  }
}

TEST(TestObserver, NullObserverIsAllowed) {
  Eigen::VectorXd x{{0.9, 0.2}};

  LevenbergMarquardt<double> solver(2);
  solver.addCost(makeSimpleCost());
  solver.setObserver(nullptr);

  const auto result = solver.optimize(x.data());

  EXPECT_GT(result.iterations, 0u);
  EXPECT_NEAR(x[0], 0.362, 0.01);
  EXPECT_NEAR(x[1], 0.556, 0.01);
}

TEST(TestObserver, StepEmitsEventsWithoutOptimize) {
  Eigen::VectorXd x{{0.9, 0.2}};

  LevenbergMarquardt<double> solver(2);
  solver.addCost(makeSimpleCost());

  RecordingObserver<double> observer;
  solver.setObserver(&observer);

  const auto status = solver.step(x.data());

  EXPECT_EQ(status, Status::STEP_OK);
  EXPECT_FALSE(observer.linear_systems.empty());
  EXPECT_FALSE(observer.iterations.empty());
}

// Illustrates hooking a plain console logger up to the optimizer. The observer
// writes to std::cout; here stdout is captured so the rendered output can be
// asserted and also echoed back for visibility.
TEST(TestObserver, ConsoleLoggingObserverIllustration) {
  Eigen::VectorXd x{{0.9, 0.2}};

  LevenbergMarquardt<double> solver(2);
  solver.addCost(makeSimpleCost());

  ConsoleLoggingObserver<double> console_logger;  // writes to std::cout by default
  solver.setObserver(&console_logger);

  testing::internal::CaptureStdout();
  const auto result = solver.optimize(x.data());
  const std::string output = testing::internal::GetCapturedStdout();

  std::cout << output;  // echo the illustration into the test log

  EXPECT_GT(result.iterations, 0u);
  EXPECT_NE(output.find("[linear system] iter 0"), std::string::npos);
  EXPECT_NE(output.find("phase ACCEPTED"), std::string::npos);
  EXPECT_NE(output.find("phase FINISHED"), std::string::npos);
}
