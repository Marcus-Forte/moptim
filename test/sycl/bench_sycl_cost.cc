// Opt-in benchmark (not run by ctest): SYCL vs CPU forward-difference cost on the same 3D point
// cloud, for a range of point counts.
//
//   build_sycl_cpu/test/bench_sycl_cost [reps]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/NumericalCostSycl.hh"
#include "transform3d.hh"

namespace {

using Clock = std::chrono::steady_clock;

/// Minimum wall time of `call` over `reps` runs, in microseconds.
template <class Call>
double minMicros(int reps, Call&& call) {
  double best = 1e300;
  for (int i = 0; i < reps; ++i) {
    const auto start = Clock::now();
    call();
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    best = std::min(best, us);
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const int max_reps = argc > 1 ? std::atoi(argv[1]) : 20;

  sycl::queue queue{sycl::default_selector_v};
  std::printf("device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("%-10s %-16s %12s %12s %9s %12s\n", "points", "operation", "cpu [us]", "sycl [us]", "cpu/sycl",
              "rel. diff");

  const double x[6]{0.1, 0.2, 0.3, 0.05, 0.02, 0.03};

  for (const size_t n : {1000ul, 10000ul, 100000ul, 1000000ul}) {
    // Same dataset for both implementations: random cloud and a rigidly transformed copy.
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    std::vector<double> source(3 * n), target(3 * n);
    for (auto& v : source) v = dist(gen);
    Point3Distance truth;
    const double x_true[6]{-0.2, 0.1, 0.05, 0.03, -0.01, 0.02};
    truth.setState(x_true);
    for (size_t i = 0; i < n; ++i) {
      Eigen::Map<const Eigen::Vector3d> s(&source[3 * i]);
      Eigen::Map<Eigen::Vector3d> t(&target[3 * i]);
      t = truth.transform_ * s;
    }

    // Fewer repetitions for the large datasets.
    const int reps = std::max(3, static_cast<int>(std::min<size_t>(max_reps, 20000000ul / n / 20 + 3)));

    moptim::NumericalCostForwardEuler<Point3Distance, double> cpu(source.data(), target.data(), n, 3, 3, 6);
    moptim::NumericalCostSycl<double, Point3Distance> sycl_cost(queue, std::span<const double>(source),
                                                                std::span<const double>(target), 3, 3, 6, n);

    // Warm up (JIT, allocations) and compare results.
    double cost_cpu = 0, cost_sycl = 0;
    Eigen::Matrix<double, 6, 6> jtj_cpu, jtj_sycl;
    Eigen::Matrix<double, 6, 1> jtb_cpu, jtb_sycl;
    cpu.computeLinearSystem(x, jtj_cpu.data(), jtb_cpu.data(), cost_cpu);
    sycl_cost.computeLinearSystem(x, jtj_sycl.data(), jtb_sycl.data(), cost_sycl);
    const double jtj_diff = (jtj_cpu - jtj_sycl).norm() / jtj_cpu.norm();

    const double cost_cpu_us = minMicros(reps, [&] { cost_cpu = cpu.computeCost(x); });
    const double cost_sycl_us = minMicros(reps, [&] { cost_sycl = sycl_cost.computeCost(x); });
    std::printf("%-10zu %-16s %12.1f %12.1f %9.2f %12.2e\n", n, "computeCost", cost_cpu_us, cost_sycl_us,
                cost_cpu_us / cost_sycl_us, std::abs(cost_cpu - cost_sycl) / cost_cpu);

    const double lin_cpu_us =
        minMicros(reps, [&] { cpu.computeLinearSystem(x, jtj_cpu.data(), jtb_cpu.data(), cost_cpu); });
    const double lin_sycl_us =
        minMicros(reps, [&] { sycl_cost.computeLinearSystem(x, jtj_sycl.data(), jtb_sycl.data(), cost_sycl); });
    std::printf("%-10zu %-16s %12.1f %12.1f %9.2f %12.2e\n", n, "linearSystem", lin_cpu_us, lin_sycl_us,
                lin_cpu_us / lin_sycl_us, jtj_diff);
    std::fflush(stdout);
  }
  return 0;
}
