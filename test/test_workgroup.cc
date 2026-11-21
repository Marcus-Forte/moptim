// The WorkGroup assembly strategy (one element per work-item, group sums in local memory), forced on
// native_cpu and compared with the Chunked strategy.
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
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"

using namespace moptim;

namespace {

constexpr size_t kInputDim = 2;
constexpr size_t kObservationDim = 2;
constexpr size_t kParamDim = 4;

// Polynomial model, two residuals and four parameters, no libm calls.
template <class T>
struct PolyModel {
  void setState(const T* /*x*/) {}
  void residual(const T* x, const T* in, const T* obs, T* res) const {
    res[0] = obs[0] - (x[0] * in[0] + x[1] * in[1] * in[1] + x[2]);
    res[1] = obs[1] - (x[3] * in[0] * in[1] + x[0] * x[1]);
  }
  void jacobian(const T* x, const T* in, const T* /*obs*/, T* jac) const {
    jac[0] = -in[0];
    jac[1] = -in[1] * in[1];
    jac[2] = T{-1};
    jac[3] = T{0};
    jac[4] = -x[1];
    jac[5] = -x[0];
    jac[6] = T{0};
    jac[7] = -in[0] * in[1];
  }
};

enum class Noise { None, Weights, Information };
const Noise kNoises[] = {Noise::None, Noise::Weights, Noise::Information};
const LossKind kLosses[] = {LossKind::Trivial, LossKind::Huber, LossKind::Cauchy, LossKind::GemanMcClure};

template <class T>
std::shared_ptr<LossFunction<T>> makeLoss(LossKind kind) {
  switch (kind) {
    case LossKind::Huber:
      return std::make_shared<HuberLoss<T>>(T{1.5});
    case LossKind::Cauchy:
      return std::make_shared<CauchyLoss<T>>(T{1.5});
    case LossKind::GemanMcClure:
      return std::make_shared<GemanMcClureLoss<T>>(T{1.5});
    default:
      return nullptr;
  }
}

std::shared_ptr<ExecutionContext> makeContext(AssemblyStrategy strategy, size_t threads = 0, size_t local_size = 0) {
  auto context = std::make_shared<ExecutionContext>(
      sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()}, threads, strategy);
  context->setWorkGroupSize(local_size);
  return context;
}

template <class T>
struct Dataset {
  std::vector<T> input, observations, weights;
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> information;
};

template <class T>
Dataset<T> makeDataset(size_t n) {
  std::mt19937 rng(29);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> weight(0.25, 4.0);
  Dataset<T> data;
  for (size_t e = 0; e < n; ++e) {
    for (size_t i = 0; i < kInputDim; ++i) data.input.push_back(static_cast<T>(value(rng)));
    for (size_t i = 0; i < kObservationDim; ++i) data.observations.push_back(static_cast<T>(value(rng)));
    data.weights.push_back(static_cast<T>(weight(rng)));
    Eigen::Matrix<T, 2, 2> factor;
    factor << static_cast<T>(value(rng)), static_cast<T>(value(rng)), static_cast<T>(value(rng)),
        static_cast<T>(value(rng));
    data.information.emplace_back(factor * factor.transpose() + Eigen::Matrix<T, 2, 2>::Identity());
  }
  return data;
}

template <class T>
struct System {
  std::vector<T> jtj, jtb;
  T system_cost{}, cost_only{};
};

template <class T>
System<T> evaluate(ICost<T>& cost, const std::vector<T>& x) {
  const size_t tangent = cost.tangentDim();
  System<T> result;
  result.jtj.assign(tangent * tangent, T{123});
  result.jtb.assign(tangent, T{-456});
  result.system_cost = T{789};
  cost.computeLinearSystem(x.data(), result.jtj.data(), result.jtb.data(), result.system_cost);
  result.cost_only = cost.computeCost(x.data());
  return result;
}

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
template <class T>
bool sameBits(T a, T b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <class T>
bool sameBits(const System<T>& a, const System<T>& b) {
  return sameBits(a.jtj, b.jtj) && sameBits(a.jtb, b.jtb) && sameBits(a.system_cost, b.system_cost) &&
         sameBits(a.cost_only, b.cost_only);
}

template <class T>
void expectClose(const System<T>& actual, const System<T>& expected, double tolerance) {
  const auto relative = [](const std::vector<T>& a, const std::vector<T>& b) {
    double diff = 0.0, norm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      diff += std::pow(static_cast<double>(a[i]) - static_cast<double>(b[i]), 2);
      norm += std::pow(static_cast<double>(b[i]), 2);
    }
    return std::sqrt(diff) / (1.0 + std::sqrt(norm));
  };
  EXPECT_LE(relative(actual.jtj, expected.jtj), tolerance);
  EXPECT_LE(relative(actual.jtb, expected.jtb), tolerance);
  EXPECT_NEAR(static_cast<double>(actual.system_cost), static_cast<double>(expected.system_cost),
              tolerance * (1.0 + std::abs(static_cast<double>(expected.system_cost))));
  EXPECT_NEAR(static_cast<double>(actual.cost_only), static_cast<double>(expected.cost_only),
              tolerance * (1.0 + std::abs(static_cast<double>(expected.cost_only))));
}

/// Builds the cost on `context` with the noise and loss, and evaluates it.
template <class T, template <class, class> class CostT>
std::unique_ptr<CostT<PolyModel<T>, T>> makeCost(const Dataset<T>& data, size_t n, size_t full_dim, Noise noise,
                                                 LossKind loss, const std::shared_ptr<ExecutionContext>& context) {
  auto cost = std::make_unique<CostT<PolyModel<T>, T>>(data.input.data(), data.observations.data(), n, kInputDim,
                                                       kObservationDim, full_dim, PolyModel<T>{}, kParamDim, context);
  if (noise == Noise::Weights) cost->setWeights(data.weights);
  if (noise == Noise::Information) cost->setInformation(data.information);
  cost->setLossFunction(makeLoss<T>(loss));
  return cost;
}

template <class T, template <class, class> class CostT>
void compareWithChunked(double tolerance) {
  const std::vector<size_t> counts = {0, 1, 5, 100, 1001, 3001};
  const size_t sizes[] = {1, 7, 64, 128};
  std::vector<T> x(kParamDim + 3, T{5});
  const T start[kParamDim] = {T{0.3}, T{-0.7}, T{0.2}, T{1.1}};
  std::copy(start, start + kParamDim, x.begin());

  for (const size_t n : counts) {
    const Dataset<T> data = makeDataset<T>(n);
    for (const Noise noise : kNoises) {
      for (const LossKind loss : kLosses) {
        const auto chunked_context = makeContext(AssemblyStrategy::Chunked);
        const System<T> expected = evaluate<T>(*makeCost<T, CostT>(data, n, x.size(), noise, loss, chunked_context), x);
        for (const size_t local : sizes) {
          SCOPED_TRACE(testing::Message() << "elements=" << n << " noise=" << static_cast<int>(noise)
                                          << " loss=" << static_cast<int>(loss) << " local=" << local);
          const auto context = makeContext(AssemblyStrategy::WorkGroup, 0, local);
          const auto cost = makeCost<T, CostT>(data, n, x.size(), noise, loss, context);
          const System<T> actual = evaluate<T>(*cost, x);
          expectClose(actual, expected, tolerance);
          EXPECT_TRUE(sameBits(actual.system_cost, actual.cost_only)) << "linear-system cost != computeCost";
          // The inactive block stays exactly zero.
          const size_t tangent = x.size();
          for (size_t a = 0; a < tangent; ++a) {
            for (size_t b = 0; b < tangent; ++b) {
              if (a >= kParamDim || b >= kParamDim) {
                EXPECT_EQ(actual.jtj[b + a * tangent], T{0});
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

TEST(WorkGroupDouble, AnalyticalMatchesChunked) { compareWithChunked<double, AnalyticalCost>(1e-12); }
TEST(WorkGroupDouble, ForwardMatchesChunked) { compareWithChunked<double, NumericalCostForwardEuler>(1e-12); }
TEST(WorkGroupDouble, CentralMatchesChunked) { compareWithChunked<double, NumericalCostCentral>(1e-12); }
TEST(WorkGroupFloat, AnalyticalMatchesChunked) { compareWithChunked<float, AnalyticalCost>(1e-4); }
TEST(WorkGroupFloat, CentralMatchesChunked) { compareWithChunked<float, NumericalCostCentral>(1e-2); }

namespace {

struct PlaneCloud {
  std::vector<double> inputs, offsets;
};

PlaneCloud makePlaneCloud(size_t n) {
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  PlaneCloud cloud;
  cloud.inputs.resize(6 * n);
  cloud.offsets.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d p(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d q = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i + 3]} = q;
    cloud.offsets[i] = q.dot(p) + 0.05 * gauss(rng);
  }
  return cloud;
}

template <template <class, class> class CostT>
void checkManifolds() {
  using namespace manifold_models;
  const std::shared_ptr<const IManifold<double>> pose = std::make_shared<SE3Manifold>();
  const std::shared_ptr<const IManifold<double>> product =
      std::make_shared<ProductManifold<double>>(std::vector<std::shared_ptr<const IManifold<double>>>{
          pose, std::make_shared<EuclideanManifold<double>>(2)});
  Eigen::Matrix<double, 6, 1> tangent;
  tangent << 0.01, -0.02, 0.03, 0.01, -0.01, 0.02;
  const Eigen::Matrix<double, 12, 1> pose_x = poseFromTangent(tangent);
  std::vector<double> product_x(pose_x.data(), pose_x.data() + 12);
  product_x.push_back(1.5);
  product_x.push_back(-2.5);

  for (const size_t n : {size_t{1}, size_t{100}, size_t{2049}}) {
    const PlaneCloud cloud = makePlaneCloud(n);
    std::vector<double> weights(n, 0.75);
    for (const size_t local : {size_t{7}, size_t{128}}) {
      SCOPED_TRACE(testing::Message() << "elements=" << n << " local=" << local);
      for (const bool use_product : {false, true}) {
        const auto& space = use_product ? product : pose;
        const std::vector<double> x = use_product ? product_x : std::vector<double>(pose_x.data(), pose_x.data() + 12);
        const auto build = [&](const std::shared_ptr<ExecutionContext>& context) {
          auto cost = std::make_unique<CostT<PointToPlane, double>>(
              cloud.inputs.data(), cloud.offsets.data(), n, 6, 1, space, PointToPlane{}, 6, context);
          cost->setWeights(weights);
          cost->setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
          return cost;
        };
        const System<double> expected = evaluate<double>(*build(makeContext(AssemblyStrategy::Chunked)), x);
        const System<double> actual =
            evaluate<double>(*build(makeContext(AssemblyStrategy::WorkGroup, 0, local)), x);
        expectClose(actual, expected, 1e-12);
        EXPECT_TRUE(sameBits(actual.system_cost, actual.cost_only));
      }
    }
  }
}

}  // namespace

TEST(WorkGroupDouble, ManifoldsAnalytical) { checkManifolds<AnalyticalCost>(); }
TEST(WorkGroupDouble, ManifoldsForward) { checkManifolds<NumericalCostForwardEuler>(); }
TEST(WorkGroupDouble, ManifoldsCentral) { checkManifolds<NumericalCostCentral>(); }

// Within the strategy: the same bits whatever the thread setting, and from run to run.
TEST(WorkGroupDeterminism, BitwiseAcrossThreadsAndRuns) {
  const Dataset<double> data = makeDataset<double>(5000);
  std::vector<double> x(kParamDim, 0.2);
  for (const size_t local : {size_t{7}, size_t{128}}) {
    System<double> first;
    bool have_first = false;
    for (const size_t threads : {size_t{1}, size_t{2}, size_t{5}, size_t{0}}) {
      for (int run = 0; run < 2; ++run) {
        SCOPED_TRACE(testing::Message() << "local=" << local << " threads=" << threads << " run=" << run);
        const auto context = makeContext(AssemblyStrategy::WorkGroup, threads, local);
        const auto cost = makeCost<double, NumericalCostCentral>(data, 5000, kParamDim, Noise::Information,
                                                                 LossKind::Cauchy, context);
        const System<double> result = evaluate<double>(*cost, x);
        EXPECT_TRUE(sameBits(result.system_cost, result.cost_only));
        if (!have_first) {
          first = result;
          have_first = true;
        } else {
          EXPECT_TRUE(sameBits(result, first));
        }
      }
    }
  }
}

TEST(WorkGroupLaunches, OnePerCallAndNoneForNoElements) {
  const Dataset<double> data = makeDataset<double>(1000);
  const std::vector<double> x(kParamDim, 0.2);
  const auto context = makeContext(AssemblyStrategy::WorkGroup, 0, 64);
  const auto cost = makeCost<double, AnalyticalCost>(data, 1000, kParamDim, Noise::Weights, LossKind::Huber, context);
  System<double> result;
  result.jtj.resize(16);
  result.jtb.resize(4);
  for (int repetition = 0; repetition < 3; ++repetition) {
    const uint64_t before = context->launches();
    cost->computeLinearSystem(x.data(), result.jtj.data(), result.jtb.data(), result.system_cost);
    EXPECT_EQ(context->launches(), before + 1);
    cost->computeCost(x.data());
    EXPECT_EQ(context->launches(), before + 2);
    // Asynchronously: a submit is the launch, wait() adds none.
    cost->submitCost(x.data(), &result.cost_only);
    EXPECT_EQ(context->launches(), before + 3);
    cost->wait();
    EXPECT_EQ(context->launches(), before + 3);
    EXPECT_TRUE(sameBits(result.cost_only, result.system_cost));
  }

  const auto empty_context = makeContext(AssemblyStrategy::WorkGroup);
  const auto empty = makeCost<double, AnalyticalCost>(data, 0, kParamDim, Noise::None, LossKind::Trivial, empty_context);
  EXPECT_EQ(empty->computeCost(x.data()), 0.0);
  empty->computeLinearSystem(x.data(), result.jtj.data(), result.jtb.data(), result.system_cost);
  EXPECT_EQ(empty_context->launches(), 0u);
}

namespace {

// A model whose residual is not const: it cannot be shared by the elements' work-items, so the cost
// uses the chunked kernel for it, whatever the strategy.
struct MutableModel {
  double scratch = 0.0;
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) {
    scratch = x[0] * in[0];
    res[0] = obs[0] - (scratch + x[1]);
  }
  void jacobian(const double* /*x*/, const double* in, const double* /*obs*/, double* jac) {
    scratch = -in[0];
    jac[0] = scratch;
    jac[1] = -1.0;
  }
};

}  // namespace

TEST(WorkGroupModels, NonConstModelFallsBackToChunked) {
  std::vector<double> in, obs;
  for (int i = 0; i < 2000; ++i) {
    in.push_back(0.001 * i);
    obs.push_back(0.3 - 0.0002 * i);
  }
  const double x[2] = {0.7, -0.3};
  double jtj_chunked[4], jtb_chunked[2], c_chunked, jtj[4], jtb[2], c;
  AnalyticalCost<MutableModel, double> chunked(in.data(), obs.data(), 2000, 1, 1, 2, MutableModel{}, 0,
                                               makeContext(AssemblyStrategy::Chunked));
  AnalyticalCost<MutableModel, double> forced(in.data(), obs.data(), 2000, 1, 1, 2, MutableModel{}, 0,
                                              makeContext(AssemblyStrategy::WorkGroup));
  chunked.computeLinearSystem(x, jtj_chunked, jtb_chunked, c_chunked);
  forced.computeLinearSystem(x, jtj, jtb, c);
  EXPECT_TRUE(sameBits(c, c_chunked));
  EXPECT_EQ(std::memcmp(jtj, jtj_chunked, sizeof(jtj)), 0);
}

// Levenberg-Marquardt reaches the same solution with either strategy.
TEST(WorkGroupOptimizer, LevenbergMarquardtMatchesChunked) {
  const Dataset<double> data = makeDataset<double>(3000);
  std::vector<std::vector<double>> solutions;
  for (const AssemblyStrategy strategy : {AssemblyStrategy::Chunked, AssemblyStrategy::WorkGroup}) {
    const auto context = makeContext(strategy, 0, 64);
    LevenbergMarquardt<double> solver(kParamDim);
    solver.addCost(makeCost<double, AnalyticalCost>(data, 3000, kParamDim, Noise::Weights, LossKind::Huber, context));
    solver.addCost(makeCost<double, NumericalCostCentral>(data, 3000, kParamDim, Noise::None, LossKind::Trivial, context));
    solver.setMaxIterations(30);
    std::vector<double> x{0.3, -0.7, 0.2, 1.1};
    const System<double> unused;
    (void)unused;
    const auto result = solver.optimize(x.data());
    EXPECT_GT(result.iterations, 0u);
    solutions.push_back(x);
  }
  for (size_t i = 0; i < kParamDim; ++i) {
    EXPECT_NEAR(solutions[0][i], solutions[1][i], 1e-6 * (1.0 + std::abs(solutions[0][i])));
  }
}

TEST(WorkGroupContext, SizeAndStrategySelection) {
  const auto context = makeContext(AssemblyStrategy::Auto);
  EXPECT_EQ(context->workGroupSize(), 0u);
  EXPECT_LE(context->resolvedWorkGroupSize(), 128u);
  EXPECT_GE(context->resolvedWorkGroupSize(), 1u);
  context->setWorkGroupSize(1000000);
  EXPECT_EQ(context->resolvedWorkGroupSize(), std::min<size_t>(1000000, context->resolvedWorkGroupSize()));
  context->setWorkGroupSize(7);
  EXPECT_EQ(context->resolvedWorkGroupSize(), 7u);
  // Auto picks WorkGroup exactly on a device that is not a CPU; the others are what they say.
  EXPECT_EQ(context->usesWorkGroupKernel(), !context->isCpu());
  context->setStrategy(AssemblyStrategy::WorkGroup);
  EXPECT_TRUE(context->usesWorkGroupKernel());
  context->setStrategy(AssemblyStrategy::Chunked);
  EXPECT_FALSE(context->usesWorkGroupKernel());
}
