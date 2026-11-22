// The linear system and the cost of an analytical cost must not depend on how many threads the
// execution context runs a launch on: the chunks are grouped in partitions that depend on the
// problem only, and the partial sums are added in partition order. Every comparison here is bitwise.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "manifold_models.hh"
#include "moptim/AnalyticalCost.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/detail/Partitioning.hh"

using namespace moptim;

namespace {

constexpr size_t kInputDim = 2;
constexpr size_t kObservationDim = 2;
constexpr size_t kParamDim = 4;

// A polynomial model with two residuals per element and four parameters (no libm calls).
template <class T>
struct PolyModel {
  void setState(const T* /*x*/) {}

  void residual(const T* x, const T* in, const T* obs, T* res) const {
    res[0] = obs[0] - (x[0] * in[0] + x[1] * in[1] * in[1] + x[2]);
    res[1] = obs[1] - (x[3] * in[0] * in[1] + x[0] * x[1]);
  }

  // jac[j + kParamDim * i] = d residual[i] / d x[j]
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

const char* name(Noise noise) {
  switch (noise) {
    case Noise::None:
      return "none";
    case Noise::Weights:
      return "weights";
    case Noise::Information:
      return "information";
  }
  return "?";
}

const char* name(LossKind kind) {
  switch (kind) {
    case LossKind::Trivial:
      return "none";
    case LossKind::Huber:
      return "huber";
    case LossKind::Cauchy:
      return "cauchy";
    case LossKind::GemanMcClure:
      return "geman-mcclure";
    case LossKind::Custom:
      return "custom";
  }
  return "?";
}

/// `Trivial` stands for "no loss set".
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

const LossKind kLosses[] = {LossKind::Trivial, LossKind::Huber, LossKind::Cauchy, LossKind::GemanMcClure};
const Noise kNoises[] = {Noise::None, Noise::Weights, Noise::Information};

// 0 stands for "all compute units".
const size_t kThreadCounts[] = {1, 2, 3, 5, 0};

/// One context per thread count, on the default backend (on SYCL, on one shared queue).
std::vector<std::shared_ptr<ExecutionContext>> makeContexts() {
  std::vector<std::shared_ptr<ExecutionContext>> contexts;
#if MOPTIM_HAS_SYCL
  if (defaultBackend() == Backend::Sycl) {
    const sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    for (const size_t threads : kThreadCounts) {
      contexts.push_back(std::make_shared<ExecutionContext>(queue, threads));
    }
    return contexts;
  }
#endif
  for (const size_t threads : kThreadCounts) {
    contexts.push_back(std::make_shared<ExecutionContext>(Backend::Host, threads));
  }
  return contexts;
}

template <class T>
struct Dataset {
  std::vector<T> input;
  std::vector<T> observations;
  std::vector<T> weights;
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> information;
};

template <class T>
Dataset<T> makeDataset(size_t num_elements) {
  std::mt19937 rng(23);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> weight(0.25, 4.0);

  Dataset<T> data;
  for (size_t e = 0; e < num_elements; ++e) {
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
struct Evaluation {
  std::vector<T> jtj;
  std::vector<T> jtb;
  T system_cost{};
  T cost_only{};
};

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <class T>
bool sameBits(T a, T b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
bool sameBits(const Evaluation<T>& a, const Evaluation<T>& b) {
  return sameBits(a.jtj, b.jtj) && sameBits(a.jtb, b.jtb) && sameBits(a.system_cost, b.system_cost) &&
         sameBits(a.cost_only, b.cost_only);
}

template <class T, class Cost>
Evaluation<T> evaluate(Cost& cost, const std::vector<T>& x) {
  const size_t tangent = cost.tangentDim();
  Evaluation<T> result;
  // Garbage in: the outputs are overwritten, never accumulated into.
  result.jtj.assign(tangent * tangent, T{123});
  result.jtb.assign(tangent, T{-456});
  result.system_cost = T{789};
  cost.computeLinearSystem(x.data(), result.jtj.data(), result.jtb.data(), result.system_cost);
  result.cost_only = cost.computeCost(x.data());
  return result;
}

/// Builds the cost for every thread count with `make(context)`, evaluates each (twice, to see that a
/// second call reuses its buffers without changing the answer) and checks that all the results are
/// bit-identical to the first and that the linear system's cost is the cost of computeCost().
template <class T, class Make>
void expectIndependentOfThreads(const Make& make, const std::vector<T>& x) {
  const auto contexts = makeContexts();
  Evaluation<T> first;
  bool have_first = false;
  for (size_t i = 0; i < contexts.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "threads=" << kThreadCounts[i]);
    const auto cost = make(contexts[i]);
    const Evaluation<T> once = evaluate<T>(*cost, x);
    const Evaluation<T> again = evaluate<T>(*cost, x);
    EXPECT_TRUE(sameBits(once, again)) << "a repeated call gave another result";
    EXPECT_TRUE(sameBits(once.system_cost, once.cost_only))
        << "linear-system cost " << once.system_cost << " computeCost " << once.cost_only;
    if (!have_first) {
      first = once;
      have_first = true;
    } else {
      EXPECT_TRUE(sameBits(once, first)) << "differs from the single-threaded result";
    }
  }
}

/// Element counts around the chunk and partition boundaries of a cost with these dimensions.
std::vector<size_t> elementCounts(size_t observation_dim, size_t active_params, size_t value_bytes) {
  const size_t chunk = detail::chunkElements(1000000, observation_dim, active_params, value_bytes);
  const size_t partition = chunk * detail::kMinChunksPerPartition;
  return {0,
          1,
          chunk - 1,
          chunk,
          chunk + 1,
          partition - 1,
          partition,
          partition + 1,
          3 * partition + chunk / 2,
          50000};
}

template <class T, template <class, class> class CostT>
void checkPolynomial(size_t full_dim) {
  const std::vector<size_t> counts = elementCounts(kObservationDim, kParamDim, sizeof(T));
  std::vector<T> x(full_dim, T{5});
  const T start[kParamDim] = {T{0.3}, T{-0.7}, T{0.2}, T{1.1}};
  std::copy(start, start + kParamDim, x.begin());

  for (const size_t num_elements : counts) {
    const Dataset<T> data = makeDataset<T>(num_elements);
    for (const Noise noise : kNoises) {
      for (const LossKind loss : kLosses) {
        SCOPED_TRACE(testing::Message() << "elements=" << num_elements << " noise=" << name(noise)
                                        << " loss=" << name(loss) << " full_dim=" << full_dim);
        const auto make = [&](const std::shared_ptr<ExecutionContext>& context) {
          auto cost = std::make_shared<CostT<PolyModel<T>, T>>(
              data.input.data(), data.observations.data(), num_elements, kInputDim, kObservationDim, full_dim,
              PolyModel<T>{}, kParamDim, context);
          if (noise == Noise::Weights) cost->setWeights(data.weights);
          if (noise == Noise::Information) cost->setInformation(data.information);
          cost->setLossFunction(makeLoss<T>(loss));
          return cost;
        };
        expectIndependentOfThreads<T>(make, x);
      }
    }
  }
}

}  // namespace

#define MOPTIM_DETERMINISM_TESTS(Suite, T, Kind, CostT)                                    \
  TEST(Suite, Polynomial##Kind) { checkPolynomial<T, CostT>(kParamDim); }                  \
  TEST(Suite, ActiveParamDimBelowParamDim##Kind) { checkPolynomial<T, CostT>(7); }

MOPTIM_DETERMINISM_TESTS(DeterminismDouble, double, Analytical, AnalyticalCost)
MOPTIM_DETERMINISM_TESTS(DeterminismFloat, float, Analytical, AnalyticalCost)
MOPTIM_DETERMINISM_TESTS(DeterminismDouble, double, Forward, NumericalCostForwardEuler)
MOPTIM_DETERMINISM_TESTS(DeterminismFloat, float, Forward, NumericalCostForwardEuler)
MOPTIM_DETERMINISM_TESTS(DeterminismDouble, double, Central, NumericalCostCentral)
MOPTIM_DETERMINISM_TESTS(DeterminismFloat, float, Central, NumericalCostCentral)

namespace {

struct PlaneCloud {
  std::vector<double> inputs;  // [p, n] per element
  std::vector<double> offsets;
};

PlaneCloud makePlaneCloud(size_t num_elements) {
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
  std::normal_distribution<double> gauss(0.0, 1.0);

  PlaneCloud cloud;
  cloud.inputs.resize(6 * num_elements);
  cloud.offsets.resize(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    const Eigen::Vector3d p(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d n = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i + 3]} = n;
    cloud.offsets[i] = n.dot(p) + 0.05 * gauss(rng);
  }
  return cloud;
}

}  // namespace

// SE(3) point-to-plane on a manifold, with weights and a loss; and the same pose inside a product
// manifold whose second block the cost does not depend on (active_param_dim 6 of 8).
template <template <class, class> class CostT>
void checkPointToPlaneOnManifolds() {
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

  const std::vector<size_t> counts = elementCounts(1, 6, sizeof(double));
  for (const size_t num_elements : counts) {
    const PlaneCloud cloud = makePlaneCloud(num_elements);
    std::vector<double> weights(num_elements);
    for (size_t e = 0; e < num_elements; ++e) weights[e] = 0.5 + 0.001 * static_cast<double>(e % 1000);

    for (const bool robust : {false, true}) {
      SCOPED_TRACE(testing::Message() << "elements=" << num_elements << " robust=" << robust);
      const auto configure = [&](auto& cost) {
        if (robust) {
          cost.setWeights(weights);
          cost.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
        }
      };
      expectIndependentOfThreads<double>(
          [&](const std::shared_ptr<ExecutionContext>& context) {
            auto cost = std::make_shared<CostT<PointToPlane, double>>(
                cloud.inputs.data(), cloud.offsets.data(), num_elements, 6, 1, pose, PointToPlane{}, 0, context);
            configure(*cost);
            return cost;
          },
          std::vector<double>(pose_x.data(), pose_x.data() + 12));
      expectIndependentOfThreads<double>(
          [&](const std::shared_ptr<ExecutionContext>& context) {
            auto cost = std::make_shared<CostT<PointToPlane, double>>(
                cloud.inputs.data(), cloud.offsets.data(), num_elements, 6, 1, product, PointToPlane{}, 6, context);
            configure(*cost);
            return cost;
          },
          product_x);
    }
  }
}

TEST(DeterminismDouble, PointToPlaneOnManifoldsAnalytical) { checkPointToPlaneOnManifolds<AnalyticalCost>(); }
TEST(DeterminismDouble, PointToPlaneOnManifoldsForward) { checkPointToPlaneOnManifolds<NumericalCostForwardEuler>(); }
TEST(DeterminismDouble, PointToPlaneOnManifoldsCentral) { checkPointToPlaneOnManifolds<NumericalCostCentral>(); }
