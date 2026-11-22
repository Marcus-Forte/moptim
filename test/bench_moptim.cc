// Benchmarks for the cost classes and optimizers on synthetic problems.
//
// Not part of the unit-test suite. Build the `bench_moptim` target and run it by hand:
//   build/test/bench_moptim            # everything
//   build/test/bench_moptim forward    # only benchmarks whose name contains "forward"
//   build/test/bench_moptim --threads 4 --strategy chunked [filter]
//   build/test/bench_moptim --speculative   # LM rows evaluate trial points with linear systems
//
// --backend host|sycl (default: MOPTIM_BACKEND, else sycl in a SYCL build), --threads N and
// --strategy auto|chunked|workgroup configure the default ExecutionContext before anything runs; the
// first line printed is the backend. One SYCL build thus benchmarks both backends, and a build without
// SYCL the host one (compare them with test/compare_bench.py).
//
// Every row reports the best time per call over several repetitions. Rows that assemble a linear
// system also print the cost and the sums of J^T J and J^T b, so two builds can be compared for
// numerical equivalence as well as for speed.
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
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

// 3D rigid alignment, residual = target - R(roll, pitch, yaw) * (source + t).
// input_dim = observation_dim = 3, param_dim = 6.
struct Point3Distance {
  Eigen::Affine3d transform_{Eigen::Affine3d::Identity()};
  Eigen::Matrix3d Rx_{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d Ry_{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_{Eigen::Vector3d::Zero()};

  void setState(const double* x) {
    t_ = Eigen::Vector3d{x[0], x[1], x[2]};
    const Eigen::AngleAxisd roll(x[3], Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch(x[4], Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw(x[5], Eigen::Vector3d::UnitZ());
    Rx_ = roll.toRotationMatrix();
    Ry_ = pitch.toRotationMatrix();
    R_ = Rx_ * Ry_ * yaw.toRotationMatrix();
    transform_.setIdentity();
    transform_.rotate(roll * pitch * yaw);
    transform_.translate(t_);
  }

  void residual(const double* /*x*/, const double* input, const double* obs, double* res) const {
    Eigen::Map<const Eigen::Vector3d> source{input};
    Eigen::Map<const Eigen::Vector3d> target{obs};
    Eigen::Map<Eigen::Vector3d> result{res};
    result = target - transform_ * source;
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) const {
    using Vector3d = Eigen::Vector3d;
    const Vector3d s = Eigen::Map<const Vector3d>(input) + t_;
    const Vector3d p = R_ * s;

    Eigen::Map<Eigen::Matrix<double, 6, 3>> jac_t(jac);
    jac_t.topRows<3>() = -R_.transpose();
    jac_t.row(3) = Vector3d(0.0, p[2], -p[1]).transpose();
    const Vector3d q = Rx_.transpose() * p;
    jac_t.row(4) = -(Rx_ * Vector3d(q[2], 0.0, -q[0])).transpose();
    const Vector3d r_vec = Ry_.transpose() * q;
    jac_t.row(5) = -(Rx_ * Ry_ * Vector3d(-r_vec[1], r_vec[0], 0.0)).transpose();
  }
};

// y = exp(a * t + b). input_dim = observation_dim = 1, param_dim = 2.
struct ExponentialModel {
  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* input, const double* obs, double* res) const {
    res[0] = obs[0] - std::exp(x[0] * input[0] + x[1]);
  }

  void jacobian(const double* x, const double* input, const double* /*obs*/, double* jac) const {
    const double value = std::exp(x[0] * input[0] + x[1]);
    jac[0] = -input[0] * value;
    jac[1] = -value;
  }
};

// One dense element, residual = A * x - obs. Mimics a single high-dimensional factor
// (num_elements = 1, observation_dim = param_dim = 24), where optimizer overhead dominates.
struct DenseFactor {
  static constexpr int kDim = 24;
  using Vector = Eigen::Matrix<double, kDim, 1>;

  Eigen::Matrix<double, kDim, kDim> A;

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* /*input*/, const double* obs, double* res) const {
    Eigen::Map<Vector>{res} = A * Eigen::Map<const Vector>{x} - Eigen::Map<const Vector>{obs};
  }
};

struct Cloud {
  std::vector<double> source;  // num_elements * 3, model input
  std::vector<double> target;  // num_elements * 3, observations
};

Cloud makeCloud(size_t num_elements) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
  std::normal_distribution<double> noise(0.0, 0.05);

  const Eigen::Affine3d pose =
      Eigen::Translation3d(0.1, -0.2, 0.3) * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ());

  Cloud cloud;
  cloud.source.resize(num_elements * 3);
  cloud.target.resize(num_elements * 3);
  for (size_t i = 0; i < num_elements; ++i) {
    const Eigen::Vector3d point(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d moved = pose * point + Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
    Eigen::Map<Eigen::Vector3d>{&cloud.source[i * 3]} = point;
    Eigen::Map<Eigen::Vector3d>{&cloud.target[i * 3]} = moved;
  }
  return cloud;
}

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

/// --speculative: every Levenberg-Marquardt row evaluates its trial points with linear systems.
bool g_speculative = false;

template <class Solver>
void configureSolver(Solver& solver) {
  solver.setSpeculativeLinearSystems(g_speculative);
}

/// Kernel launches one `call()` issues on the default context.
template <class Call>
unsigned long long launchesOf(Call&& call) {
  const std::shared_ptr<ExecutionContext> context = defaultContext();
  const uint64_t before = context->launches();
  call();
  return static_cast<unsigned long long>(context->launches() - before);
}

/// Best wall time of one `call()`, in nanoseconds.
template <class Call>
double nsPerCall(Call&& call) {
  call();  // warm-up: page in the buffers and prime the caches

  double best = std::numeric_limits<double>::infinity();
  for (int repetition = 0; repetition < 7; ++repetition) {
    size_t calls = 0;
    const auto start = Clock::now();
    Clock::duration elapsed{};
    do {
      call();
      ++calls;
      elapsed = Clock::now() - start;
    } while (elapsed < std::chrono::milliseconds(40));
    best = std::min(best, std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(calls));
  }
  return best;
}

class Report {
 public:
  explicit Report(std::string filter) : filter_(std::move(filter)) {
    std::printf("%-46s %9s %13s %10s  %s\n", "benchmark", "elements", "time/call", "ns/elem", "notes");
  }

  bool wants(const std::string& name) const { return name.find(filter_) != std::string::npos; }

  void row(const std::string& name, size_t num_elements, double ns, const std::string& notes = "") const {
    std::printf("%-46s %9zu %10.2f us %10.2f  %s\n", name.c_str(), num_elements, ns / 1e3,
                ns / static_cast<double>(num_elements), notes.c_str());
    std::fflush(stdout);
  }

 private:
  std::string filter_;
};

template <class... Args>
std::string format(const char* pattern, Args... args) {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer), pattern, args...);
  return buffer;
}

constexpr size_t kInputDim = 3;
constexpr size_t kObservationDim = 3;
constexpr size_t kParamDim = 6;

Eigen::VectorXd linearizationPoint() {
  Eigen::VectorXd x(kParamDim);
  x << 0.01, -0.02, 0.03, 0.01, -0.01, 0.02;
  return x;
}

/// Times computeLinearSystem() and computeCost() of `cost` and reports both.
template <class Cost>
void benchCost(const Report& report, const std::string& name, Cost& cost, size_t num_elements) {
  const Eigen::VectorXd x = linearizationPoint();
  Eigen::MatrixXd jtj(kParamDim, kParamDim);
  Eigen::VectorXd jtb(kParamDim);
  double value = 0.0;

  if (report.wants("linear_system/" + name)) {
    const double ns = nsPerCall([&] {
      cost.computeLinearSystem(x.data(), jtj.data(), jtb.data(), value);
      g_sink = g_sink + value + jtj(0, 0);
    });
    report.row("linear_system/" + name, num_elements, ns,
               format("cost=%.12g sum(JTJ)=%.12g sum(JTb)=%.12g", value, jtj.sum(), jtb.sum()));
  }

  if (report.wants("cost/" + name)) {
    const double ns = nsPerCall([&] {
      value = cost.computeCost(x.data());
      g_sink = g_sink + value;
    });
    report.row("cost/" + name, num_elements, ns, format("cost=%.12g", value));
  }
}

void benchCloudCosts(const Report& report, size_t num_elements) {
  const Cloud cloud = makeCloud(num_elements);
  const double* source = cloud.source.data();
  const double* target = cloud.target.data();

  {
    AnalyticalCost<Point3Distance, double> cost(source, target, num_elements, kInputDim, kObservationDim, kParamDim);
    benchCost(report, "analytical", cost, num_elements);
  }
  {
    NumericalCostForwardEuler<Point3Distance, double> cost(source, target, num_elements, kInputDim, kObservationDim,
                                                           kParamDim);
    benchCost(report, "forward", cost, num_elements);
  }
  {
    NumericalCostCentral<Point3Distance, double> cost(source, target, num_elements, kInputDim, kObservationDim,
                                                      kParamDim);
    benchCost(report, "central", cost, num_elements);
  }

  std::mt19937 rng(11);
  {
    // Scalar weights and a Huber kernel: the per-element robust path.
    std::uniform_real_distribution<double> weight(0.5, 2.0);
    std::vector<double> weights(num_elements);
    for (double& w : weights) w = weight(rng);

    NumericalCostForwardEuler<Point3Distance, double> cost(source, target, num_elements, kInputDim, kObservationDim,
                                                           kParamDim);
    cost.setWeights(weights);
    cost.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
    benchCost(report, "forward+weights+huber", cost, num_elements);
  }
  {
    // Full per-element information matrices.
    std::uniform_real_distribution<double> entry(-1.0, 1.0);
    std::vector<Eigen::MatrixXd> information(num_elements);
    for (auto& matrix : information) {
      Eigen::Matrix3d factor;
      for (int i = 0; i < 9; ++i) factor(i) = entry(rng);
      matrix = factor * factor.transpose() + Eigen::Matrix3d::Identity();
    }

    AnalyticalCost<Point3Distance, double> cost(source, target, num_elements, kInputDim, kObservationDim, kParamDim);
    cost.setInformation(information);
    benchCost(report, "analytical+information", cost, num_elements);
  }
}

/// Cost construction followed by a single linear system, the pattern of a registration loop that
/// rebuilds its cost for every new set of correspondences.
void benchConstruction(const Report& report, size_t num_elements) {
  const std::string name = "construct+linear_system/forward";
  if (!report.wants(name)) return;

  const Cloud cloud = makeCloud(num_elements);
  const Eigen::VectorXd x = linearizationPoint();
  Eigen::MatrixXd jtj(kParamDim, kParamDim);
  Eigen::VectorXd jtb(kParamDim);
  double value = 0.0;

  const double ns = nsPerCall([&] {
    NumericalCostForwardEuler<Point3Distance, double> cost(cloud.source.data(), cloud.target.data(), num_elements,
                                                           kInputDim, kObservationDim, kParamDim);
    cost.computeLinearSystem(x.data(), jtj.data(), jtb.data(), value);
    g_sink = g_sink + value;
  });
  report.row(name, num_elements, ns);
}

template <class Cost>
void benchOptimize(const Report& report, const std::string& name, size_t num_elements) {
  if (!report.wants(name)) return;

  const Cloud cloud = makeCloud(num_elements);
  auto cost = std::make_shared<Cost>(cloud.source.data(), cloud.target.data(), num_elements, kInputDim, kObservationDim,
                                     kParamDim);
  LevenbergMarquardt<double> solver(kParamDim);
  configureSolver(solver);
  solver.addCost(cost);

  Eigen::VectorXd x(kParamDim);
  Result<double> result;
  const auto run = [&] {
    x.setZero();
    result = solver.optimize(x.data());
    g_sink = g_sink + x[0];
  };
  const auto launches = launchesOf(run);
  const double ns = nsPerCall(run);
  report.row(name, num_elements, ns,
             format("iterations=%zu status=%d final_cost=%.12g x0=%.9g launches=%llu", result.iterations,
                    static_cast<int>(result.status), result.final_cost, x[0], launches));
}

/// The inner loop of a scan-registration front end: every outer iteration rebuilds the cost for a
/// fresh correspondence set and runs a few Levenberg-Marquardt iterations on it.
void benchRegistrationLoop(const Report& report, size_t num_elements) {
  const std::string name = "registration_loop/forward (10 x 5 it)";
  if (!report.wants(name)) return;

  const Cloud cloud = makeCloud(num_elements);
  LevenbergMarquardt<double> solver(kParamDim);
  configureSolver(solver);
  solver.setMaxIterations(5);

  Eigen::VectorXd x(kParamDim);
  const auto run = [&] {
    x.setZero();
    for (int outer = 0; outer < 10; ++outer) {
      solver.clearCosts();
      solver.addCost(std::make_shared<NumericalCostForwardEuler<Point3Distance, double>>(
          cloud.source.data(), cloud.target.data(), num_elements, kInputDim, kObservationDim, kParamDim));
      solver.optimize(x.data());
    }
    g_sink = g_sink + x[0];
  };
  const auto launches = launchesOf(run);
  const double ns = nsPerCall(run);
  report.row(name, num_elements, ns, format("x0=%.9g launches=%llu", x[0], launches));
}

/// Tiny problem (67 elements, 2 parameters): dominated by per-call overhead, not by the element loop.
void benchCurveFit(const Report& report) {
  constexpr size_t kNumElements = 67;
  std::vector<double> input(kNumElements);
  std::vector<double> observations(kNumElements);
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 0.15);
  for (size_t i = 0; i < kNumElements; ++i) {
    input[i] = 0.075 * static_cast<double>(i);
    observations[i] = std::exp(0.3 * input[i] + 0.1) + noise(rng);
  }

  const auto run = [&](const std::string& name, auto make_cost) {
    if (!report.wants(name)) return;
    Eigen::VectorXd x(2);
    Result<double> result;
    const auto call = [&] {
      x.setZero();
      LevenbergMarquardt<double> solver(2);
      configureSolver(solver);
      solver.addCost(make_cost());
      result = solver.optimize(x.data());
      g_sink = g_sink + x[0];
    };
    const auto launches = launchesOf(call);
    const double ns = nsPerCall(call);
    report.row(name, kNumElements, ns,
               format("iterations=%zu status=%d final_cost=%.12g x0=%.9g launches=%llu", result.iterations,
                      static_cast<int>(result.status), result.final_cost, x[0], launches));
  };

  run("lm_optimize/curve_fit/analytical", [&] {
    return std::make_shared<AnalyticalCost<ExponentialModel, double>>(input.data(), observations.data(), kNumElements,
                                                                      1, 1, 2);
  });
  run("lm_optimize/curve_fit/forward", [&] {
    return std::make_shared<NumericalCostForwardEuler<ExponentialModel, double>>(input.data(), observations.data(),
                                                                                 kNumElements, 1, 1, 2);
  });
}

/// A single 24-dimensional factor differentiated numerically.
void benchDenseFactor(const Report& report) {
  const std::string name = "lm_optimize/dense_factor/central";
  if (!report.wants(name)) return;

  constexpr int kDim = DenseFactor::kDim;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> entry(-1.0, 1.0);

  DenseFactor model;
  for (int i = 0; i < kDim * kDim; ++i) model.A(i) = entry(rng);
  model.A.diagonal().array() += 4.0;

  DenseFactor::Vector observation;
  for (int i = 0; i < kDim; ++i) observation[i] = entry(rng);
  const DenseFactor::Vector unused_input = DenseFactor::Vector::Zero();

  Eigen::VectorXd x(kDim);
  Result<double> result;
  const auto call = [&] {
    x.setZero();
    LevenbergMarquardt<double> solver(kDim);
    configureSolver(solver);
    solver.addCost(std::make_shared<NumericalCostCentral<DenseFactor, double>>(unused_input.data(), observation.data(),
                                                                               1, kDim, kDim, kDim, model));
    result = solver.optimize(x.data());
    g_sink = g_sink + x[0];
  };
  const auto launches = launchesOf(call);
  const double ns = nsPerCall(call);
  report.row(name, 1, ns,
             format("iterations=%zu status=%d final_cost=%.12g x0=%.9g launches=%llu", result.iterations,
                    static_cast<int>(result.status), result.final_cost, x[0], launches));
}

// ─── Manifold path: SE(3) point-to-plane ───────────────────────────────────────────────────────
// x is the pose itself (12 numbers, ambient), the linear system is 6 x 6 (tangent). The analytical
// model gives the jacobian with respect to a left increment at zero; the numerical row steps the
// tangent dimensions through the manifold's plus.

struct PlaneCloud {
  std::vector<double> inputs;  // [p, n] per element
  std::vector<double> offsets;
};

PlaneCloud makePlaneCloud(size_t num_elements) {
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, 0.05);

  Eigen::Matrix<double, 6, 1> truth_tangent;
  truth_tangent << 0.1, -0.2, 0.3, 0.05, -0.03, 0.08;
  const Eigen::Matrix<double, 12, 1> truth = manifold_models::poseFromTangent(truth_tangent);
  const Eigen::Matrix3d R = Eigen::Map<const Eigen::Matrix3d>(truth.data());
  const Eigen::Vector3d t = Eigen::Map<const Eigen::Vector3d>(truth.data() + 9);

  PlaneCloud cloud;
  cloud.inputs.resize(6 * num_elements);
  cloud.offsets.resize(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    const Eigen::Vector3d p(coordinate(rng), coordinate(rng), coordinate(rng));
    const Eigen::Vector3d n = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i]} = p;
    Eigen::Map<Eigen::Vector3d>{&cloud.inputs[6 * i + 3]} = n;
    cloud.offsets[i] = n.dot(R * p + t) + noise(rng);
  }
  return cloud;
}

constexpr size_t kPoseTangentDim = 6;

template <class Cost>
void benchManifoldLinearSystem(const Report& report, const std::string& name, const PlaneCloud& cloud,
                               size_t num_elements, const std::shared_ptr<const IManifold<double>>& manifold) {
  if (!report.wants(name)) return;

  Cost cost(cloud.inputs.data(), cloud.offsets.data(), num_elements, 6, 1, manifold);
  Eigen::Matrix<double, 6, 1> start_tangent;
  start_tangent << 0.01, -0.02, 0.03, 0.01, -0.01, 0.02;
  const Eigen::Matrix<double, 12, 1> x = manifold_models::poseFromTangent(start_tangent);
  Eigen::MatrixXd jtj(kPoseTangentDim, kPoseTangentDim);
  Eigen::VectorXd jtb(kPoseTangentDim);
  double value = 0.0;

  const double ns = nsPerCall([&] {
    cost.computeLinearSystem(x.data(), jtj.data(), jtb.data(), value);
    g_sink = g_sink + value + jtj(0, 0);
  });
  report.row(name, num_elements, ns, format("cost=%.12g sum(JTJ)=%.12g sum(JTb)=%.12g", value, jtj.sum(), jtb.sum()));
}

template <class Cost>
void benchManifoldOptimize(const Report& report, const std::string& name, const PlaneCloud& cloud, size_t num_elements,
                           const std::shared_ptr<const IManifold<double>>& manifold) {
  if (!report.wants(name)) return;

  auto cost = std::make_shared<Cost>(cloud.inputs.data(), cloud.offsets.data(), num_elements, 6, 1, manifold);
  LevenbergMarquardt<double> solver(manifold);
  configureSolver(solver);
  solver.addCost(cost);

  Eigen::Matrix<double, 12, 1> x;
  Result<double> result;
  const auto call = [&] {
    x = manifold_models::poseFromTangent(Eigen::Matrix<double, 6, 1>::Zero());
    result = solver.optimize(x.data());
    g_sink = g_sink + x[9];
  };
  const auto launches = launchesOf(call);
  const double ns = nsPerCall(call);
  report.row(name, num_elements, ns,
             format("iterations=%zu status=%d final_cost=%.12g t0=%.9g launches=%llu", result.iterations,
                    static_cast<int>(result.status), result.final_cost, x[9], launches));
}

// ─── mslam's pattern: a point-to-plane cloud and a one-element prior on the same pose ─────────────
// Per scan mslam clears the costs, builds an analytical point-to-plane cost of about 1200 elements and
// a one-element numerical factor (its IMU term) on the same SE(3) manifold, and runs 5 iterations,
// 25 times. The prior here pulls the translation to a measured value.

struct TranslationPrior {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* /*input*/, const double* obs, double* res) const {
    for (int i = 0; i < 3; ++i) res[i] = obs[i] - x[9 + i];
  }
};

constexpr size_t kScanElements = 1200;

Eigen::Matrix<double, 12, 1> perturbedPose() {
  Eigen::Matrix<double, 6, 1> tangent;
  tangent << 0.02, -0.03, 0.04, 0.01, -0.02, 0.015;
  return manifold_models::poseFromTangent(tangent);
}

void addScanCosts(LevenbergMarquardt<double>& solver, const PlaneCloud& cloud, const double* prior,
                  const std::shared_ptr<const IManifold<double>>& manifold) {
  solver.addCost(std::make_shared<AnalyticalCost<manifold_models::PointToPlane, double>>(
      cloud.inputs.data(), cloud.offsets.data(), kScanElements, 6, 1, manifold));
  static const double unused_input = 0.0;  // read by the cost on every evaluation
  solver.addCost(std::make_shared<NumericalCostCentral<TranslationPrior, double>>(&unused_input, prior, 1, 1, 3,
                                                                                  manifold));
}

void benchTwoCosts(const Report& report) {
  const std::string optimize_name = "lm_optimize/two_costs";
  const std::string loop_name = "registration_loop/two_costs (25 x 5 it)";
  if (!report.wants(optimize_name) && !report.wants(loop_name)) return;

  const std::shared_ptr<const IManifold<double>> manifold = std::make_shared<manifold_models::SE3Manifold>();
  const PlaneCloud cloud = makePlaneCloud(kScanElements);
  const double prior[3] = {0.1, -0.2, 0.3};
  const Eigen::Matrix<double, 12, 1> start = perturbedPose();

  if (report.wants(optimize_name)) {
    LevenbergMarquardt<double> solver(manifold);
    configureSolver(solver);
    solver.setMaxIterations(5);
    addScanCosts(solver, cloud, prior, manifold);

    Eigen::Matrix<double, 12, 1> x;
    Result<double> result;
    const auto call = [&] {
      x = start;
      result = solver.optimize(x.data());
      g_sink = g_sink + x[9];
    };
    const auto launches = launchesOf(call);
    const double ns = nsPerCall(call);
    report.row(optimize_name, kScanElements, ns,
               format("iterations=%zu status=%d final_cost=%.12g t0=%.9g launches=%llu", result.iterations,
                      static_cast<int>(result.status), result.final_cost, x[9], launches));
  }

  if (report.wants(loop_name)) {
    LevenbergMarquardt<double> solver(manifold);
    configureSolver(solver);
    solver.setMaxIterations(5);

    Eigen::Matrix<double, 12, 1> x;
    const auto call = [&] {
      x = start;
      for (int scan = 0; scan < 25; ++scan) {
        solver.clearCosts();
        addScanCosts(solver, cloud, prior, manifold);
        solver.optimize(x.data());
      }
      g_sink = g_sink + x[9];
    };
    const auto launches = launchesOf(call);
    const double ns = nsPerCall(call);
    report.row(loop_name, kScanElements, ns, format("t0=%.9g launches=%llu", x[9], launches));
  }
}

void benchManifold(const Report& report) {
  const std::shared_ptr<const IManifold<double>> manifold = std::make_shared<manifold_models::SE3Manifold>();
  for (const size_t num_elements : {size_t{1000}, size_t{100000}}) {
    const PlaneCloud cloud = makePlaneCloud(num_elements);
    benchManifoldLinearSystem<AnalyticalCost<manifold_models::PointToPlane, double>>(
        report, "linear_system/manifold_se3/analytical", cloud, num_elements, manifold);
    benchManifoldLinearSystem<NumericalCostCentral<manifold_models::PointToPlane, double>>(
        report, "linear_system/manifold_se3/central", cloud, num_elements, manifold);
  }
  const PlaneCloud cloud = makePlaneCloud(10000);
  benchManifoldOptimize<AnalyticalCost<manifold_models::PointToPlane, double>>(
      report, "lm_optimize/manifold_se3/analytical", cloud, 10000, manifold);
  benchManifoldOptimize<NumericalCostCentral<manifold_models::PointToPlane, double>>(
      report, "lm_optimize/manifold_se3/central", cloud, 10000, manifold);
}

struct Options {
  std::string filter;
  bool configure_context = false;
  size_t threads = 0;
  AssemblyStrategy strategy = AssemblyStrategy::Auto;
  Backend backend = defaultBackend();
  bool speculative = false;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  const auto value = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--threads") {
      const std::string text = value(i);
      size_t used = 0;
      const unsigned long long threads = std::stoull(text, &used);
      if (used != text.size()) throw std::invalid_argument("bad --threads value: " + text);
      options.threads = static_cast<size_t>(threads);
      options.configure_context = true;
    } else if (arg == "--backend") {
      const std::string text = value(i);
      if (text == "host") options.backend = Backend::Host;
      else if (text == "sycl" && syclAvailable()) options.backend = Backend::Sycl;
      else throw std::invalid_argument("--backend must be host" + std::string(syclAvailable() ? " or sycl" : "") + ", not " + text);
      options.configure_context = true;
    } else if (arg == "--speculative") {
      options.speculative = true;
    } else if (arg == "--strategy") {
      const std::string text = value(i);
      if (text == "auto") options.strategy = AssemblyStrategy::Auto;
      else if (text == "chunked") options.strategy = AssemblyStrategy::Chunked;
      else if (text == "workgroup") options.strategy = AssemblyStrategy::WorkGroup;
      else throw std::invalid_argument("--strategy must be auto, chunked or workgroup, not " + text);
      options.configure_context = true;
    } else if (arg.rfind("--", 0) == 0) {
      throw std::invalid_argument("unknown option " + arg);
    } else {
      options.filter = arg;
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parseOptions(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\nusage: bench_moptim [--backend host|sycl] [--threads N] [--strategy auto|chunked|workgroup] [--speculative] [name-filter]\n",
                 error.what());
    return 2;
  }
  g_speculative = options.speculative;
  // Only touch the default context when asked: creating it initializes the SYCL runtime.
  if (options.configure_context) {
    setDefaultContext(std::make_shared<ExecutionContext>(options.backend, options.threads, options.strategy));
  }
  std::printf("backend=%s\n", backendName(defaultContext()->backend()));

  const Report report(options.filter);

  for (const size_t num_elements : {size_t{1000}, size_t{10000}, size_t{100000}, size_t{1000000}}) {
    benchCloudCosts(report, num_elements);
  }

  benchConstruction(report, 20000);
  benchOptimize<AnalyticalCost<Point3Distance, double>>(report, "lm_optimize/analytical", 10000);
  benchOptimize<NumericalCostForwardEuler<Point3Distance, double>>(report, "lm_optimize/forward", 10000);
  benchOptimize<NumericalCostCentral<Point3Distance, double>>(report, "lm_optimize/central", 10000);
  benchRegistrationLoop(report, 20000);
  benchCurveFit(report);
  benchDenseFactor(report);
  benchManifold(report);
  benchTwoCosts(report);

  return 0;
}
