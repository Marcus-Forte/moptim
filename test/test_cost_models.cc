// How AnalyticalCost treats the model it evaluates in a launch (possibly on several threads): const and
// non-const member functions, move-only models, fixed-size Eigen members, references to host data, the
// lifetime of the copies, and the launches an evaluation costs.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"

using namespace moptim;

namespace {

constexpr size_t kParams = 2;

std::shared_ptr<ExecutionContext> makeContext(size_t threads) {
  return std::make_shared<ExecutionContext>(threads);
}

struct Data {
  std::vector<double> input;
  std::vector<double> observations;
};

Data makeData(size_t n) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  Data data;
  for (size_t i = 0; i < n; ++i) {
    data.input.push_back(value(rng));
    data.observations.push_back(value(rng));
  }
  return data;
}

// residual = obs - (a * x0 * in + x1), one residual and two parameters per element.
struct LineReference {
  Eigen::Matrix2d jtj = Eigen::Matrix2d::Zero();
  Eigen::Vector2d jtb = Eigen::Vector2d::Zero();
  double cost = 0.0;
};

LineReference lineReference(const Data& data, const double* x, double a) {
  LineReference ref;
  for (size_t i = 0; i < data.input.size(); ++i) {
    const double r = data.observations[i] - (a * x[0] * data.input[i] + x[1]);
    const Eigen::Vector2d j(-a * data.input[i], -1.0);
    ref.jtj += j * j.transpose();
    ref.jtb += j * r;
    ref.cost += r * r;
  }
  return ref;
}

template <class Cost>
void expectMatchesLine(Cost& cost, const Data& data, double a, double tolerance = 1e-11) {
  const double x[kParams] = {0.7, -0.3};
  const LineReference ref = lineReference(data, x, a);
  Eigen::Matrix2d jtj;
  Eigen::Vector2d jtb;
  double value = 0.0;
  cost.computeLinearSystem(x, jtj.data(), jtb.data(), value);
  EXPECT_LE((jtj - ref.jtj).norm(), tolerance * (1.0 + ref.jtj.norm()));
  EXPECT_LE((jtb - ref.jtb).norm(), tolerance * (1.0 + ref.jtb.norm()));
  EXPECT_NEAR(value, ref.cost, tolerance * (1.0 + ref.cost));
  EXPECT_NEAR(cost.computeCost(x), ref.cost, tolerance * (1.0 + ref.cost));
}

// Enough elements for many partitions (a chunk is a few hundred elements here, a partition 4 chunks): 15, so
// that even a cost-only launch (wide from 12 partitions) is wide with three threads.
constexpr size_t kManyElements = 40000;

// ─── Non-const member functions ──────────────────────────────────────────────────────────────────

// residual() and jacobian() are not const and the model writes to itself: every work-item must
// work on its own copy.
struct ScratchModel {
  double a = 1.0;
  double scratch[2] = {0.0, 0.0};

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* in, const double* obs, double* res) {
    scratch[0] = a * x[0] * in[0];
    scratch[1] = x[1];
    res[0] = obs[0] - (scratch[0] + scratch[1]);
  }

  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) {
    scratch[0] = -a * in[0];
    jac[0] = scratch[0];
    jac[1] = -1.0;
  }
};

}  // namespace

TEST(CostModels, NonConstModelIsCopiedPerWorkItem) {
  const Data data = makeData(kManyElements);
  for (const size_t threads : {size_t{1}, size_t{3}, size_t{0}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    ScratchModel model;
    model.a = 1.25;
    AnalyticalCost<ScratchModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1, 1,
                                              kParams, model, 0, makeContext(threads));
    expectMatchesLine(cost, data, 1.25);
  }
}

namespace {

// ─── Move-only models ────────────────────────────────────────────────────────────────────────────

// Not copyable, const-callable: shared in place by every work-item.
struct MoveOnlyConstModel {
  explicit MoveOnlyConstModel(double a) : a_(std::make_unique<double>(a)) {}

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (*a_ * x[0] * in[0] + x[1]);
  }

  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -*a_ * in[0];
    jac[1] = -1.0;
  }

  std::unique_ptr<double> a_;
};

// Not copyable and not const-callable: a single work-item uses it in place, however many threads the
// context has.
struct MoveOnlyMutableModel {
  explicit MoveOnlyMutableModel(double a) : a_(std::make_unique<double>(a)) {}

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* in, const double* obs, double* res) {
    last_ = x[0] * in[0];
    res[0] = obs[0] - (*a_ * last_ + x[1]);
  }

  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) {
    last_ = in[0];
    jac[0] = -*a_ * last_;
    jac[1] = -1.0;
  }

  std::unique_ptr<double> a_;
  double last_ = 0.0;
};

static_assert(!std::is_copy_constructible_v<MoveOnlyConstModel>);
static_assert(!std::is_copy_constructible_v<MoveOnlyMutableModel>);

}  // namespace

TEST(CostModels, MoveOnlyConstModelIsShared) {
  const Data data = makeData(kManyElements);
  AnalyticalCost<MoveOnlyConstModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1, 1,
                                                  kParams, MoveOnlyConstModel(0.8), 0, makeContext(0));
  expectMatchesLine(cost, data, 0.8);
}

TEST(CostModels, MoveOnlyMutableModelRunsOnOneWorkItem) {
  const Data data = makeData(kManyElements);
  AnalyticalCost<MoveOnlyMutableModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1,
                                                    1, kParams, MoveOnlyMutableModel(0.8), 0, makeContext(0));
  expectMatchesLine(cost, data, 0.8);
}

namespace {

// ─── Fixed-size Eigen members ────────────────────────────────────────────────────────────────────

// 4x4 and 4-vector members set from x on the host and read by the work-items.
struct EigenMembersModel {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  Eigen::Vector4d offset = Eigen::Vector4d::Zero();
  double tag = 0.0;
  Eigen::Matrix<float, 3, 3> unused = Eigen::Matrix<float, 3, 3>::Zero();

  void setState(const double* x) {
    transform.setIdentity();
    transform(0, 0) = x[0];
    transform(1, 1) = 2.0;
    offset << x[1], 0.0, 0.0, 0.0;
    tag = x[0];
  }

  void residual(const double* /*x*/, const double* in, const double* obs, double* res) const {
    const Eigen::Vector4d p(in[0], 1.0, 0.0, 0.0);
    const Eigen::Vector4d q = transform * p + offset;
    res[0] = obs[0] - (q[0] + q[1] - 2.0);
  }

  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -in[0];
    jac[1] = -1.0;
  }
};

}  // namespace

TEST(CostModels, EigenMembers) {
  const Data data = makeData(5000);
  AnalyticalCost<EigenMembersModel, double> cost(data.input.data(), data.observations.data(), 5000, 1, 1, kParams,
                                                 EigenMembersModel{}, 0, makeContext(0));
  expectMatchesLine(cost, data, 1.0);
}

namespace {

// ─── A reference to host data ────────────────────────────────────────────────────────────────────

class ReferenceModel {
 public:
  explicit ReferenceModel(const std::vector<double>& table) : table_(table) {}

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (table_[0] * x[0] * in[0] + x[1]);
  }

  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -table_[0] * in[0];
    jac[1] = -1.0;
  }

 private:
  const std::vector<double>& table_;
};

}  // namespace

TEST(CostModels, ModelHoldingAReferenceToHostData) {
  const std::vector<double> table{1.5};
  const Data data = makeData(kManyElements);
  AnalyticalCost<ReferenceModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1, 1,
                                              kParams, ReferenceModel(table), 0, makeContext(0));
  expectMatchesLine(cost, data, 1.5);
}

namespace {

// ─── Constructor and destructor counts ───────────────────────────────────────────────────────────

struct Counters {
  int constructed = 0;  // every constructor that ran: default, move and copy
  int copied = 0;
  int destroyed = 0;
};

// The copies of a launch are made by several threads at once.
void bump(int& counter) {
  std::atomic_ref<int>(counter).fetch_add(1, std::memory_order_relaxed);
}

struct CountingBase {
  explicit CountingBase(Counters* c) : counters(c) { bump(counters->constructed); }
  CountingBase(CountingBase&& other) noexcept : counters(other.counters) { bump(counters->constructed); }
  CountingBase(const CountingBase& other) : counters(other.counters) {
    bump(counters->constructed);
    bump(counters->copied);
  }
  ~CountingBase() { bump(counters->destroyed); }

  void setState(const double* /*x*/) {}

  Counters* counters;
};

struct ConstCountingModel : CountingBase {
  using CountingBase::CountingBase;
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] * in[0] + x[1]);
  }
  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -in[0];
    jac[1] = -1.0;
  }
};

struct MutableCountingModel : CountingBase {
  using CountingBase::CountingBase;
  void residual(const double* x, const double* in, const double* obs, double* res) {
    res[0] = obs[0] - (x[0] * in[0] + x[1]);
  }
  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) {
    jac[0] = -in[0];
    jac[1] = -1.0;
  }
};

}  // namespace

TEST(CostModels, EveryConstructedModelIsDestroyedOnce) {
  const Data data = makeData(kManyElements);
  Counters counters;
  const auto context = makeContext(3);
  // Work-items of the launch: kManyElements has many partitions, so three threads give three work-items
  // (one without oneTBB).
  const int work_groups = ExecutionContext::parallelAvailable() ? 3 : 1;
  {
    AnalyticalCost<MutableCountingModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1,
                                                      1, kParams, MutableCountingModel(&counters), 0, context);
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;

    cost.computeLinearSystem(x, jtj, jtb, value);
    const int after_one = counters.copied;
    // Each work-item copies the model once per launch.
    EXPECT_EQ(after_one, work_groups);
    EXPECT_EQ(counters.destroyed, counters.constructed - 1) << "all the copies of the launch are gone, the model lives";

    cost.computeCost(x);
    EXPECT_EQ(counters.copied, 2 * after_one);
    EXPECT_EQ(counters.destroyed, counters.constructed - 1);
  }
  EXPECT_GT(counters.constructed, 0);
  EXPECT_EQ(counters.destroyed, counters.constructed);
}

TEST(CostModels, ConstModelIsNeverCopied) {
  const Data data = makeData(kManyElements);
  Counters counters;
  {
    AnalyticalCost<ConstCountingModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1, 1,
                                                    kParams, ConstCountingModel(&counters), 0, makeContext(3));
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;
    cost.computeLinearSystem(x, jtj, jtb, value);
    cost.computeCost(x);
    EXPECT_EQ(counters.copied, 0);
  }
  EXPECT_EQ(counters.destroyed, counters.constructed);
}

TEST(CostModels, ModelOfACostThatNeverEvaluatesIsDestroyed) {
  Counters counters;
  {
    const Data data = makeData(10);
    AnalyticalCost<ConstCountingModel, double> cost(data.input.data(), data.observations.data(), 10, 1, 1, kParams,
                                                    ConstCountingModel(&counters), 0, makeContext(1));
  }
  EXPECT_EQ(counters.destroyed, counters.constructed);
}

// ─── Launches, no elements, losses, strategy ─────────────────────────────────────────────────────

namespace {

struct LineModel {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] * in[0] + x[1]);
  }
  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -in[0];
    jac[1] = -1.0;
  }
};

}  // namespace

TEST(CostModels, OneLaunchPerEvaluation) {
  const auto context = makeContext(0);
  for (const size_t n : {size_t{1}, size_t{7}, size_t{1500}, kManyElements}) {
    SCOPED_TRACE(testing::Message() << "elements=" << n);
    const Data data = makeData(n);
    AnalyticalCost<LineModel, double> cost(data.input.data(), data.observations.data(), n, 1, 1, kParams,
                                           LineModel{}, 0, context);
    cost.setWeights(std::vector<double>(n, 2.0));
    cost.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;

    // The first evaluation allocates the buffers and checks the model's layout; neither is a launch.
    for (int repetition = 0; repetition < 3; ++repetition) {
      const uint64_t before = context->launches();
      cost.computeLinearSystem(x, jtj, jtb, value);
      EXPECT_EQ(context->launches(), before + 1);
      cost.computeCost(x);
      EXPECT_EQ(context->launches(), before + 2);
    }
  }
}

TEST(CostModels, NoElementsNoLaunch) {
  const auto context = makeContext(0);
  AnalyticalCost<LineModel, double> cost(nullptr, nullptr, 0, 1, 1, kParams, LineModel{}, 0, context);
  const double x[kParams] = {0.7, -0.3};
  double jtj[kParams * kParams] = {1, 2, 3, 4};
  double jtb[kParams] = {5, 6};
  double value = 9.0;

  cost.computeLinearSystem(x, jtj, jtb, value);
  EXPECT_EQ(context->launches(), 0u);
  EXPECT_EQ(value, 0.0);
  for (const double entry : jtj) EXPECT_EQ(entry, 0.0);
  for (const double entry : jtb) EXPECT_EQ(entry, 0.0);
  EXPECT_EQ(cost.computeCost(x), 0.0);
  EXPECT_EQ(context->launches(), 0u);
}

namespace {

class CustomLoss : public LossFunction<double> {
 public:
  void evaluate(double s, double out[3]) const override {
    out[0] = s;
    out[1] = 1.0;
    out[2] = 0.0;
  }
};

}  // namespace

TEST(CostModels, CustomLossIsRejected) {
  const Data data = makeData(10);
  AnalyticalCost<LineModel, double> cost(data.input.data(), data.observations.data(), 10, 1, 1, kParams, LineModel{},
                                         0, makeContext(1));
  EXPECT_THROW(cost.setLossFunction(std::make_shared<CustomLoss>()), std::invalid_argument);
  // The built-ins and removing the loss are fine.
  EXPECT_NO_THROW(cost.setLossFunction(std::make_shared<TrivialLoss<double>>()));
  EXPECT_NO_THROW(cost.setLossFunction(nullptr));
}

// The descriptor is read at every evaluation, so a scale changed in between is seen.
TEST(CostModels, LossScaleChangesTakeEffect) {
  const Data data = makeData(2000);
  const double x[kParams] = {0.7, -0.3};
  double jtj[kParams * kParams];
  double jtb[kParams];

  const auto loss = std::make_shared<HuberLoss<double>>(5.0);
  AnalyticalCost<LineModel, double> adapted(data.input.data(), data.observations.data(), 2000, 1, 1, kParams,
                                            LineModel{}, 0, makeContext(2));
  adapted.setLossFunction(loss);
  const double wide = adapted.computeCost(x);

  loss->setScale(0.1);
  const double narrow = adapted.computeCost(x);
  EXPECT_LT(narrow, wide);

  AnalyticalCost<LineModel, double> fresh(data.input.data(), data.observations.data(), 2000, 1, 1, kParams,
                                          LineModel{}, 0, makeContext(2));
  fresh.setLossFunction(std::make_shared<HuberLoss<double>>(0.1));
  EXPECT_EQ(narrow, fresh.computeCost(x));

  double cost = 0.0;
  adapted.computeLinearSystem(x, jtj, jtb, cost);
  EXPECT_EQ(cost, narrow);
}

// ─── Numerical costs: one model per evaluation point ─────────────────────────────────────────────

namespace {
// Finite differences of a function that is linear in the parameters are exact to a few 1e-9.
constexpr double kNumericalTolerance = 1e-5;
}  // namespace

TEST(NumericalCostModels, NonConstModelIsCopiedPerWorkItem) {
  const Data data = makeData(kManyElements);
  for (const size_t threads : {size_t{1}, size_t{3}, size_t{0}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    ScratchModel model;
    model.a = 1.25;
    NumericalCostCentral<ScratchModel, double> central(data.input.data(), data.observations.data(), kManyElements, 1,
                                                       1, kParams, model, 0, makeContext(threads));
    expectMatchesLine(central, data, 1.25, kNumericalTolerance);
    NumericalCostForwardEuler<ScratchModel, double> forward(data.input.data(), data.observations.data(),
                                                            kManyElements, 1, 1, kParams, model, 0,
                                                            makeContext(threads));
    expectMatchesLine(forward, data, 1.25, kNumericalTolerance);
  }
}

TEST(NumericalCostModels, EigenMembers) {
  const Data data = makeData(5000);
  NumericalCostCentral<EigenMembersModel, double> cost(data.input.data(), data.observations.data(), 5000, 1, 1,
                                                       kParams, EigenMembersModel{}, 0, makeContext(0));
  expectMatchesLine(cost, data, 1.0, kNumericalTolerance);
}

TEST(NumericalCostModels, ModelHoldingAReferenceToHostData) {
  const std::vector<double> table{1.5};
  const Data data = makeData(kManyElements);
  NumericalCostForwardEuler<ReferenceModel, double> cost(data.input.data(), data.observations.data(), kManyElements,
                                                         1, 1, kParams, ReferenceModel(table), 0, makeContext(0));
  expectMatchesLine(cost, data, 1.5, kNumericalTolerance);
}

// A cost keeps 1 + P_active models (forward) or 1 + 2 P_active (central), built once; a model that is not
// const-callable is also copied once per work-item and launch (all of them for a linear system, the
// first for the cost alone), and every copy is destroyed.
template <template <class, class> class CostT, size_t kPointsPerParam>
void checkModelCounts() {
  constexpr size_t kModels = 1 + kPointsPerParam * kParams;
  const auto context = makeContext(3);
  // Work-items of a launch: three threads give three work-items for this many partitions (one without oneTBB).
  const int kWorkItems = ExecutionContext::parallelAvailable() ? 3 : 1;
  const Data data = makeData(kManyElements);
  Counters counters;
  const auto alive = [&] { return counters.constructed - counters.destroyed; };
  {
    CostT<MutableCountingModel, double> cost(data.input.data(), data.observations.data(), kManyElements, 1, 1,
                                             kParams, MutableCountingModel(&counters), 0, context);
    EXPECT_EQ(alive(), 1) << "nothing is built before the first evaluation";
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;

    cost.computeLinearSystem(x, jtj, jtb, value);
    EXPECT_EQ(alive(), static_cast<int>(kModels));
    EXPECT_EQ(counters.copied, static_cast<int>(kModels - 1) + kWorkItems * static_cast<int>(kModels));

    const int copied = counters.copied;
    cost.computeCost(x);
    EXPECT_EQ(alive(), static_cast<int>(kModels));
    EXPECT_EQ(counters.copied, copied + kWorkItems);

    cost.computeLinearSystem(x, jtj, jtb, value);
    EXPECT_EQ(alive(), static_cast<int>(kModels)) << "no model is built again";
  }
  EXPECT_EQ(counters.destroyed, counters.constructed);
}

TEST(NumericalCostModels, EveryConstructedModelIsDestroyedOnceForward) { checkModelCounts<NumericalCostForwardEuler, 1>(); }
TEST(NumericalCostModels, EveryConstructedModelIsDestroyedOnceCentral) { checkModelCounts<NumericalCostCentral, 2>(); }

TEST(NumericalCostModels, ConstModelIsNeverCopiedPerWorkItem) {
  const Data data = makeData(kManyElements);
  Counters counters;
  {
    NumericalCostCentral<ConstCountingModel, double> cost(data.input.data(), data.observations.data(), kManyElements,
                                                          1, 1, kParams, ConstCountingModel(&counters), 0,
                                                          makeContext(3));
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;
    cost.computeLinearSystem(x, jtj, jtb, value);
    cost.computeCost(x);
    EXPECT_EQ(counters.copied, 2 * static_cast<int>(kParams));  // the 2 P extra models, once
  }
  EXPECT_EQ(counters.destroyed, counters.constructed);
}

TEST(NumericalCostModels, OneLaunchPerEvaluation) {
  const auto context = makeContext(0);
  for (const size_t n : {size_t{1}, size_t{7}, size_t{1500}, kManyElements}) {
    SCOPED_TRACE(testing::Message() << "elements=" << n);
    const Data data = makeData(n);
    NumericalCostCentral<LineModel, double> cost(data.input.data(), data.observations.data(), n, 1, 1, kParams,
                                                 LineModel{}, 0, context);
    cost.setWeights(std::vector<double>(n, 2.0));
    cost.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
    const double x[kParams] = {0.7, -0.3};
    double jtj[kParams * kParams];
    double jtb[kParams];
    double value = 0.0;
    for (int repetition = 0; repetition < 3; ++repetition) {
      const uint64_t before = context->launches();
      cost.computeLinearSystem(x, jtj, jtb, value);
      EXPECT_EQ(context->launches(), before + 1);
      cost.computeCost(x);
      EXPECT_EQ(context->launches(), before + 2);
    }
  }
}

TEST(NumericalCostModels, NoElementsNoLaunch) {
  const auto context = makeContext(0);
  NumericalCostForwardEuler<LineModel, double> cost(nullptr, nullptr, 0, 1, 1, kParams, LineModel{}, 0, context);
  const double x[kParams] = {0.7, -0.3};
  double jtj[kParams * kParams] = {1, 2, 3, 4};
  double jtb[kParams] = {5, 6};
  double value = 9.0;
  cost.computeLinearSystem(x, jtj, jtb, value);
  EXPECT_EQ(cost.computeCost(x), 0.0);
  EXPECT_EQ(context->launches(), 0u);
  EXPECT_EQ(value, 0.0);
  for (const double entry : jtj) EXPECT_EQ(entry, 0.0);
}

TEST(NumericalCostModels, CustomLossIsRejected) {
  const Data data = makeData(10);
  NumericalCostCentral<LineModel, double> cost(data.input.data(), data.observations.data(), 10, 1, 1, kParams,
                                               LineModel{}, 0, makeContext(1));
  EXPECT_THROW(cost.setLossFunction(std::make_shared<CustomLoss>()), std::invalid_argument);
}

// ─── The jacobian scatter of AnalyticalCost for each observation dimension ─────────────────────
// AnalyticalJacobian::evaluateJacobian() has code instantiated for dimensions 1, 2 and 3 and a generic one for
// the others: each is checked against the jacobian written out here, so a mixed-up row or column shows.

// residual i of an element = obs[i] - ((i + 1) * x0 * in + (i + 2) * x1), `kDim` residuals, two parameters.
template <int kDim>
struct RowsModel {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* input, const double* obs, double* res) const {
    for (int i = 0; i < kDim; ++i) {
      res[i] = obs[i] - ((i + 1) * x[0] * input[0] + (i + 2) * x[1]);
    }
  }
  // The transposed jacobian: entry (p, i) at p + 2 * i.
  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    for (int i = 0; i < kDim; ++i) {
      jac[0 + 2 * i] = -(i + 1) * input[0];
      jac[1 + 2 * i] = -(i + 2);
    }
  }
};

template <int kDim>
void expectScatterMatchesReference(size_t elements) {
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::vector<double> input(elements);
  std::vector<double> observations(elements * kDim);
  for (double& v : input) v = value(rng);
  for (double& v : observations) v = value(rng);

  const double x[2] = {0.7, -0.3};
  Eigen::Matrix2d expected_jtj = Eigen::Matrix2d::Zero();
  Eigen::Vector2d expected_jtb = Eigen::Vector2d::Zero();
  double expected_cost = 0.0;
  for (size_t e = 0; e < elements; ++e) {
    for (int i = 0; i < kDim; ++i) {
      const double r = observations[e * kDim + i] - ((i + 1) * x[0] * input[e] + (i + 2) * x[1]);
      const Eigen::Vector2d j(-(i + 1) * input[e], -(i + 2));
      expected_jtj += j * j.transpose();
      expected_jtb += j * r;
      expected_cost += r * r;
    }
  }

  AnalyticalCost<RowsModel<kDim>, double> cost(input.data(), observations.data(), elements, 1, kDim, 2,
                                               RowsModel<kDim>{}, 0, makeContext(1));
  Eigen::Matrix2d jtj;
  Eigen::Vector2d jtb;
  double actual_cost = 0.0;
  cost.computeLinearSystem(x, jtj.data(), jtb.data(), actual_cost);

  constexpr double kTolerance = 1e-10;
  EXPECT_LE((jtj - expected_jtj).norm(), kTolerance * (1.0 + expected_jtj.norm())) << "dim " << kDim;
  EXPECT_LE((jtb - expected_jtb).norm(), kTolerance * (1.0 + expected_jtb.norm())) << "dim " << kDim;
  EXPECT_NEAR(actual_cost, expected_cost, kTolerance * (1.0 + expected_cost)) << "dim " << kDim;
}

TEST(AnalyticalJacobianScatter, MatchesReferenceForEveryObservationDimension) {
  // 3000 elements: several chunks, and a last chunk that is not full.
  expectScatterMatchesReference<1>(3000);
  expectScatterMatchesReference<2>(3000);
  expectScatterMatchesReference<3>(3000);
  expectScatterMatchesReference<4>(3000);  // the generic path (dimension known at run time)
  expectScatterMatchesReference<7>(3000);
  expectScatterMatchesReference<2>(1);  // a single element
}
