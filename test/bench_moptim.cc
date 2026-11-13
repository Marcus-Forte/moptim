// Benchmarks for the cost classes and optimizers on synthetic problems.
//
// Not part of the unit-test suite. Build the `bench_moptim` target and run it by hand:
//   build/test/bench_moptim            # everything
//   build/test/bench_moptim forward    # only benchmarks whose name contains "forward"
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
#include <string>
#include <vector>

#include "moptim/AnalyticalCost.hh"
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
  solver.addCost(cost);

  Eigen::VectorXd x(kParamDim);
  Result<double> result;
  const double ns = nsPerCall([&] {
    x.setZero();
    result = solver.optimize(x.data());
    g_sink = g_sink + x[0];
  });
  report.row(name, num_elements, ns,
             format("iterations=%zu status=%d final_cost=%.12g x0=%.9g", result.iterations,
                    static_cast<int>(result.status), result.final_cost, x[0]));
}

/// The inner loop of a scan-registration front end: every outer iteration rebuilds the cost for a
/// fresh correspondence set and runs a few Levenberg-Marquardt iterations on it.
void benchRegistrationLoop(const Report& report, size_t num_elements) {
  const std::string name = "registration_loop/forward (10 x 5 it)";
  if (!report.wants(name)) return;

  const Cloud cloud = makeCloud(num_elements);
  LevenbergMarquardt<double> solver(kParamDim);
  solver.setMaxIterations(5);

  Eigen::VectorXd x(kParamDim);
  const double ns = nsPerCall([&] {
    x.setZero();
    for (int outer = 0; outer < 10; ++outer) {
      solver.clearCosts();
      solver.addCost(std::make_shared<NumericalCostForwardEuler<Point3Distance, double>>(
          cloud.source.data(), cloud.target.data(), num_elements, kInputDim, kObservationDim, kParamDim));
      solver.optimize(x.data());
    }
    g_sink = g_sink + x[0];
  });
  report.row(name, num_elements, ns, format("x0=%.9g", x[0]));
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
    const double ns = nsPerCall([&] {
      x.setZero();
      LevenbergMarquardt<double> solver(2);
      solver.addCost(make_cost());
      result = solver.optimize(x.data());
      g_sink = g_sink + x[0];
    });
    report.row(name, kNumElements, ns,
               format("iterations=%zu status=%d final_cost=%.12g x0=%.9g", result.iterations,
                      static_cast<int>(result.status), result.final_cost, x[0]));
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
  const double ns = nsPerCall([&] {
    x.setZero();
    LevenbergMarquardt<double> solver(kDim);
    solver.addCost(std::make_shared<NumericalCostCentral<DenseFactor, double>>(unused_input.data(), observation.data(),
                                                                               1, kDim, kDim, kDim, model));
    result = solver.optimize(x.data());
    g_sink = g_sink + x[0];
  });
  report.row(name, 1, ns,
             format("iterations=%zu status=%d final_cost=%.12g x0=%.9g", result.iterations,
                    static_cast<int>(result.status), result.final_cost, x[0]));
}

}  // namespace

int main(int argc, char** argv) {
  const Report report(argc > 1 ? argv[1] : "");

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

  return 0;
}
