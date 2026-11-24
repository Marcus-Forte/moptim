// ExecutionContext: how wide a launch is, where its work-items run (inline, or on the oneTBB arena),
// that an exception of a worker reaches the caller, and that the thread count never changes a result.
// The assertions about several threads are skipped in a build without oneTBB.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/FusedCost.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/detail/AlignedArray.hh"
#include "moptim/detail/Partitioning.hh"

using namespace moptim;

namespace {

/// r = exp(a u) + b - y, with (a, b) read from the state; counts its copies.
struct CountedExpModel {
  static inline int copies = 0;
  double a = 0.0;
  double b = 0.0;

  CountedExpModel() = default;
  CountedExpModel(const CountedExpModel& other) : a(other.a), b(other.b) { ++copies; }
  CountedExpModel(CountedExpModel&&) = default;
  CountedExpModel& operator=(const CountedExpModel&) = default;

  void setState(const double* x) {
    a = x[0];
    b = x[1];
  }
  void residual(const double* /*x*/, const double* input, const double* observation, double* r) const {
    r[0] = std::exp(a * input[0]) + b - observation[0];
  }
};

/// A line r = obs - (x0 u + x1) whose residual is not const: every work-item copies it, so the copies of
/// an evaluation are the work-items it ran on. Optionally throws, to see an exception leave a worker.
struct CopyCountingLine {
  static inline std::atomic<int> copies{0};
  static inline std::atomic<bool> fail{false};
  double scratch = 0.0;

  CopyCountingLine() = default;
  CopyCountingLine(const CopyCountingLine& other) : scratch(other.scratch) { copies.fetch_add(1); }
  CopyCountingLine(CopyCountingLine&&) = default;

  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) {
    if (fail.load(std::memory_order_relaxed) && in[0] > 1.5) {
      throw std::runtime_error("residual failed");
    }
    scratch = x[0] * in[0];
    res[0] = obs[0] - (scratch + x[1]);
  }
  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) {
    jac[0] = -in[0];
    jac[1] = -1.0;
  }
};

struct Line {
  std::vector<double> input;
  std::vector<double> observations;
};

/// `n` elements with inputs spread over [0, 2], so the last ones are those that make a failing model throw.
Line makeLine(size_t n) {
  Line line;
  for (size_t i = 0; i < n; ++i) {
    const double u = 2.0 * static_cast<double>(i) / static_cast<double>(n);
    line.input.push_back(u);
    line.observations.push_back(0.5 * u + 0.25 + 0.01 * std::sin(static_cast<double>(i)));
  }
  return line;
}

bool sameBits(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

/// Elements for which a cost of one residual and two parameters has `partitions` partitions at least.
size_t elementsForPartitions(size_t partitions) {
  const size_t chunk = detail::chunkElements(1000000, 1, 2, sizeof(double));
  return partitions * detail::kMinChunksPerPartition * chunk;
}

size_t partitionsOf(size_t elements) { return detail::makePartitioning(elements, 1, 2, sizeof(double)).num_partitions; }
}  // namespace

TEST(ExecutionContextTest, EveryWorkGroupRunsOnceAtAnyThreadCount) {
  for (const size_t threads : {size_t{1}, size_t{2}, size_t{3}, size_t{0}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    ExecutionContext context(threads);
    for (const size_t groups : {size_t{1}, size_t{5}, size_t{37}}) {
      std::vector<int> counts(groups, 0);
      int* data = counts.data();
      const uint64_t before = context.launches();
      detail::submit(context, groups, [data](size_t group) { std::atomic_ref<int>(data[group]).fetch_add(1); });
      EXPECT_EQ(context.launches() - before, 1u);
      for (size_t g = 0; g < groups; ++g) {
        ASSERT_EQ(counts[g], 1) << g;
      }
    }
    const uint64_t before = context.launches();
    detail::submit(context, 0, [](size_t) { FAIL() << "no work-group to run"; });
    EXPECT_EQ(context.launches(), before) << "a launch of nothing is not counted";
  }
}

TEST(ExecutionContextTest, ThreadsAndResolvedThreads) {
  ExecutionContext all;
  EXPECT_EQ(all.threads(), 0u);
  EXPECT_GE(all.resolvedThreads(), 1u);
  ExecutionContext three(3);
  EXPECT_EQ(three.threads(), 3u);
  EXPECT_EQ(three.resolvedThreads(), ExecutionContext::parallelAvailable() ? 3u : 1u);
  three.setThreads(1);
  EXPECT_EQ(three.threads(), 1u);
  EXPECT_EQ(three.resolvedThreads(), 1u);
  three.setThreads(0);
  EXPECT_EQ(three.resolvedThreads(), all.resolvedThreads());
  EXPECT_EQ(defaultContext()->threads(), 0u);
}

// Inline below a minimum number of partitions (6 for a linear system, 12 for a cost-only launch: below that
// waking the workers costs more than it saves), then at most one work-item per two partitions.
TEST(ExecutionContextTest, LaunchWidth) {
  ExecutionContext context(8);
  EXPECT_EQ(context.launchWidth(0), 0u);
  EXPECT_EQ(context.launchWidth(0, LaunchWork::CostOnly), 0u);
  if (!ExecutionContext::parallelAvailable()) {
    for (const size_t partitions : {size_t{1}, size_t{5}, size_t{12}, size_t{1000}}) {
      EXPECT_EQ(context.launchWidth(partitions), 1u);
      EXPECT_EQ(context.launchWidth(partitions, LaunchWork::CostOnly), 1u);
    }
    return;
  }
  for (const size_t partitions : {size_t{1}, size_t{2}, size_t{3}, size_t{5}}) {
    EXPECT_EQ(context.launchWidth(partitions), 1u) << partitions;
  }
  EXPECT_EQ(context.launchWidth(6), 3u);
  EXPECT_EQ(context.launchWidth(7), 4u);
  EXPECT_EQ(context.launchWidth(13), 7u);
  EXPECT_EQ(context.launchWidth(1000), 8u);
  for (const size_t partitions : {size_t{1}, size_t{6}, size_t{11}}) {
    EXPECT_EQ(context.launchWidth(partitions, LaunchWork::CostOnly), 1u) << partitions;
  }
  EXPECT_EQ(context.launchWidth(12, LaunchWork::CostOnly), 6u);
  EXPECT_EQ(context.launchWidth(13, LaunchWork::CostOnly), 7u);
  EXPECT_EQ(context.launchWidth(1000, LaunchWork::CostOnly), 8u);
  context.setThreads(1);
  EXPECT_EQ(context.launchWidth(1000), 1u);
  EXPECT_EQ(context.launchWidth(1000, LaunchWork::CostOnly), 1u);
  context.setThreads(2);
  EXPECT_EQ(context.launchWidth(1000), 2u);
}

TEST(ExecutionContextTest, OneWorkGroupAndOneThreadRunOnTheCallingThread) {
  const auto caller = std::this_thread::get_id();
  const auto on_caller = [caller](size_t) { EXPECT_EQ(std::this_thread::get_id(), caller); };
  ExecutionContext several(4);
  detail::submit(several, 1, on_caller);
  ExecutionContext one(1);
  detail::submit(one, 9, on_caller);
}

TEST(ExecutionContextTest, WorkGroupsRunOnSeveralThreads) {
  if (!ExecutionContext::parallelAvailable() || std::thread::hardware_concurrency() < 2) {
    GTEST_SKIP() << "needs oneTBB and two hardware threads";
  }
  ExecutionContext context(4);
  std::mutex mutex;
  std::set<std::thread::id> seen;
  // Each work-group waits (bounded) until another thread showed up, so that a launch that ran
  // everything on one thread is noticed without the test depending on the scheduler's speed.
  const auto record = [&](size_t) {
    {
      const std::lock_guard lock(mutex);
      seen.insert(std::this_thread::get_id());
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      {
        const std::lock_guard lock(mutex);
        if (seen.size() > 1) {
          return;
        }
      }
      std::this_thread::yield();
    }
  };
  detail::submit(context, 16, record);
  EXPECT_GT(seen.size(), 1u);
}

TEST(ExecutionContextTest, AnExceptionOfAWorkGroupReachesTheCaller) {
  for (const size_t threads : {size_t{1}, size_t{4}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    ExecutionContext context(threads);
    auto throwing = [](size_t group) {
      if (group == 11) {
        throw std::runtime_error("work-group 11");
      }
    };
    EXPECT_THROW(detail::runKernelOnHost(context, 16, throwing), std::runtime_error);
    // The context is still usable.
    std::atomic<int> total{0};
    auto counting = [&total](size_t) { total.fetch_add(1); };
    detail::runKernelOnHost(context, 16, counting);
    EXPECT_EQ(total.load(), 16);
  }
}

// A model that throws in a worker: the evaluation throws in the caller, and the cost is usable after.
TEST(ExecutionContextTest, AnExceptionOfAModelInAWorkerReachesTheCaller) {
  const size_t n = elementsForPartitions(12);
  const Line line = makeLine(n);
  const auto context = std::make_shared<ExecutionContext>(4);
  AnalyticalCost<CopyCountingLine, double> cost(line.input.data(), line.observations.data(), n, 1, 1, 2,
                                                CopyCountingLine{}, 0, context);
  const double x[2] = {0.5, 0.25};
  double jtj[4], jtb[2], value = 0.0;
  const double expected = cost.computeCost(x);

  CopyCountingLine::fail = true;
  EXPECT_THROW(cost.computeCost(x), std::runtime_error);
  EXPECT_THROW(cost.computeLinearSystem(x, jtj, jtb, value), std::runtime_error);
  CopyCountingLine::fail = false;

  EXPECT_TRUE(sameBits(&expected, std::array<double, 1>{cost.computeCost(x)}.data(), sizeof(double)));
  cost.computeLinearSystem(x, jtj, jtb, value);
  EXPECT_TRUE(sameBits(&expected, &value, sizeof(double)));
}

// The work-items of an evaluation: many partitions get several when the context has threads, a cost below
// the minimum stays on one (a linear system from 6 partitions, a cost-only launch from 12), and one thread
// means one.
TEST(ExecutionContextTest, CostsGetTheLaunchWidthOfTheirPartitions) {
  const double x[2] = {0.5, 0.25};
  const auto work_items = [&](size_t elements, size_t threads, bool linear_system) {
    const Line line = makeLine(elements);
    const auto context = std::make_shared<ExecutionContext>(threads);
    AnalyticalCost<CopyCountingLine, double> cost(line.input.data(), line.observations.data(), elements, 1, 1, 2,
                                                  CopyCountingLine{}, 0, context);
    double jtj[4], jtb[2], value = 0.0;
    CopyCountingLine::copies = 0;
    if (linear_system) {
      cost.computeLinearSystem(x, jtj, jtb, value);
    } else {
      cost.computeCost(x);
    }
    return CopyCountingLine::copies.load();
  };
  const bool parallel = ExecutionContext::parallelAvailable();
  const auto elements_for = [](size_t partitions) {
    const size_t elements = elementsForPartitions(partitions);
    EXPECT_EQ(partitionsOf(elements), partitions);
    return elements;
  };

  for (const bool linear_system : {true, false}) {
    const size_t minimum = linear_system ? 6 : 12;
    SCOPED_TRACE(linear_system ? "linear system" : "cost only");
    const size_t many = elements_for(minimum);
    for (const size_t threads : {size_t{1}, size_t{2}, size_t{3}, size_t{5}}) {
      const size_t expected = parallel ? std::min(threads, (minimum + 1) / 2) : 1;
      EXPECT_EQ(work_items(many, threads, linear_system), static_cast<int>(expected)) << "threads=" << threads;
    }
    if (parallel) {
      EXPECT_GT(work_items(many, 2, linear_system), 1);
    }
    EXPECT_EQ(work_items(many, 1, linear_system), 1);
    // One partition fewer: inline whatever the threads.
    EXPECT_EQ(work_items(elements_for(minimum - 1), 8, linear_system), 1);
    EXPECT_EQ(work_items(elements_for(1), 8, linear_system), 1);
    EXPECT_EQ(work_items(elements_for(2), 8, linear_system), 1);
  }
  // Between the two minima a linear system is wide and a cost alone is not.
  EXPECT_EQ(work_items(elements_for(8), 8, true), parallel ? 4 : 1);
  EXPECT_EQ(work_items(elements_for(8), 8, false), 1);
}

// Every evaluation is one launch, whatever its width, and a FusedCost is one launch too.
TEST(ExecutionContextTest, OneLaunchPerEvaluationAtAnyWidth) {
  const size_t n = elementsForPartitions(12);
  const Line line = makeLine(n);
  const auto context = std::make_shared<ExecutionContext>(4);
  using Cost = AnalyticalCost<CopyCountingLine, double>;
  const auto a =
      std::make_shared<Cost>(line.input.data(), line.observations.data(), n, 1, 1, 2, CopyCountingLine{}, 0, context);
  const auto b =
      std::make_shared<Cost>(line.input.data(), line.observations.data(), n, 1, 1, 2, CopyCountingLine{}, 0, context);
  const double x[2] = {0.5, 0.25};
  double jtj[4], jtb[2], value = 0.0;
  uint64_t before = context->launches();
  a->computeLinearSystem(x, jtj, jtb, value);
  a->computeCost(x);
  EXPECT_EQ(context->launches() - before, 2u);

  const auto fused = makeFusedCost<double>(a, b);
  before = context->launches();
  fused->computeLinearSystem(x, jtj, jtb, value);
  fused->computeCost(x);
  EXPECT_EQ(context->launches() - before, 2u);
}

TEST(ExecutionContextTest, AlignedMemory) {
  auto aligned = detail::makeAlignedArray<double>(5, 128);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned.get()) % 128, 0u);
  auto empty = detail::makeAlignedArray<double>(0);
  EXPECT_NE(empty.get(), nullptr);
  auto flags = detail::makeAlignedArray<float>(16);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(flags.get()) % detail::kDefaultAlignment, 0u);
}

// A numerical cost whose elements fit in one chunk visits each evaluation point once, so it
// moves its one model from point to point instead of copying it per point.
// A FusedCost runs the per-point copies: the results are the same bits.
TEST(ExecutionContextTest, ANumericalCostInOneChunkCopiesNoModel) {
  const auto context = std::make_shared<ExecutionContext>(4);
  std::vector<double> input, observations;
  for (int e = 0; e < 40; ++e) {
    input.push_back(-1.0 + 0.05 * e);
    observations.push_back(std::exp(0.4 * input.back()) - 0.1 + 0.01 * std::sin(3.0 * e));
  }
  using Central = NumericalCostCentral<CountedExpModel, double>;
  const double x[2] = {0.3, -0.2};

  CountedExpModel::copies = 0;
  Central alone(input.data(), observations.data(), 40, 1, 1, 2, CountedExpModel{}, 0, context);
  double jtj[4], jtb[2], cost = 0.0;
  const uint64_t before = context->launches();
  alone.computeLinearSystem(x, jtj, jtb, cost);
  const double cost_only = alone.computeCost(x);
  EXPECT_EQ(context->launches() - before, 2u);
  EXPECT_EQ(CountedExpModel::copies, 0);

  CountedExpModel::copies = 0;
  const auto fused = makeFusedCost<double>(
      std::make_shared<Central>(input.data(), observations.data(), 40, 1, 1, 2, CountedExpModel{}, 0, context));
  double fused_jtj[4], fused_jtb[2], fused_cost = 0.0;
  fused->computeLinearSystem(x, fused_jtj, fused_jtb, fused_cost);
  EXPECT_EQ(CountedExpModel::copies, 4);  // two points per parameter
  EXPECT_TRUE(sameBits(jtj, fused_jtj, sizeof(jtj)));
  EXPECT_TRUE(sameBits(jtb, fused_jtb, sizeof(jtb)));
  EXPECT_TRUE(sameBits(&cost, &fused_cost, sizeof(cost)));
  EXPECT_TRUE(sameBits(&cost, &cost_only, sizeof(cost)));
}
