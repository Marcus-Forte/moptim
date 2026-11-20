// Phase 0 spike: the analytical chunk loop of bench_moptim's Point3Distance problem, run either
// directly on the calling thread (--mode host) or by SYCL work-items (--mode kernel).
//
//   spike_chunk --mode host
//   spike_chunk --mode kernel --threads 4 [--partitions single|auto]
//
// Prints bench_moptim's row format and timing method so test/compare_bench.py can compare the
// output with a bench_moptim run. The native_cpu thread pool size comes from the environment
// (SYCL_NATIVE_CPU_HOST_THREADS), not from --threads, which is the number of work-groups launched.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "spike_kernel.hh"
#include "spike_model.hh"

namespace spike {
bool secondTuSelfCheck();
}

namespace {

using namespace spike;
using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

// Copied from bench_moptim.cc: best wall time of one `call()`, in nanoseconds.
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

void printRow(const std::string& name, size_t num_elements, double ns, const std::string& notes) {
  std::printf("%-46s %9zu %10.2f us %10.2f  %s\n", name.c_str(), num_elements, ns / 1e3,
              ns / static_cast<double>(num_elements), notes.c_str());
  std::fflush(stdout);
}

[[noreturn]] void usage() {
  std::fprintf(stderr, "usage: spike --mode host|kernel [--threads W] [--partitions single|auto]\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "kernel";
  size_t threads = 1;
  auto partitions = LinearSystem<Point3Distance>::Partitions::Auto;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc) usage();
    const std::string value = argv[++i];
    if (arg == "--mode" && (value == "host" || value == "kernel")) {
      mode = value;
    } else if (arg == "--threads" && std::atol(value.c_str()) > 0) {
      threads = static_cast<size_t>(std::atol(value.c_str()));
    } else if (arg == "--partitions" && (value == "single" || value == "auto")) {
      partitions = value == "single" ? LinearSystem<Point3Distance>::Partitions::Single
                                     : LinearSystem<Point3Distance>::Partitions::Auto;
    } else {
      usage();
    }
  }
  const bool host = mode == "host";
  // Host mode is the codegen reference: one partition covering all elements.
  if (host) partitions = LinearSystem<Point3Distance>::Partitions::Single;

  if (!secondTuSelfCheck()) {
    std::fprintf(stderr, "second TU self-check FAILED\n");
    return 1;
  }

  sycl::queue queue{sycl::default_selector_v, {sycl::property::queue::in_order()}};
  std::fprintf(stderr, "mode=%s threads=%zu partitions=%s device=%s\n", mode.c_str(), threads,
               partitions == LinearSystem<Point3Distance>::Partitions::Single ? "single" : "auto",
               queue.get_device().get_info<sycl::info::device::name>().c_str());

  std::printf("%-46s %9s %13s %10s  %s\n", "benchmark", "elements", "time/call", "ns/elem", "notes");
  const double x[kParamDim] = {0.01, -0.02, 0.03, 0.01, -0.01, 0.02};

  for (const size_t n : {size_t{1000}, size_t{10000}, size_t{100000}, size_t{1000000}}) {
    const Cloud cloud = makeCloud(n);
    LinearSystem<Point3Distance> system(queue, cloud.source, cloud.target, threads, partitions);
    Result result;
    const double ns = nsPerCall([&] {
      if (host) {
        system.runHost(x, result);
      } else {
        system.runKernel(x, threads, result);
      }
      g_sink = g_sink + result.cost + result.hessian[0];
    });

    double sum_jtj = 0.0, sum_jtb = 0.0;
    for (size_t i = 0; i < kParamDim * kParamDim; ++i) sum_jtj += result.hessian[i];
    for (size_t i = 0; i < kParamDim; ++i) sum_jtb += result.gradient[i];
    char notes[256];
    std::snprintf(notes, sizeof(notes), "cost=%.12g sum(JTJ)=%.12g sum(JTb)=%.12g", result.cost, sum_jtj, sum_jtb);
    // Hash of the exact bits of the result, for comparing runs bit for bit (stderr, not part of the rows).
    unsigned long long hash = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&result);
    for (size_t i = 0; i < sizeof(Result); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    std::fprintf(stderr, "bits n=%zu fnv=%016llx\n", n, hash);
    printRow("linear_system/analytical", n, ns, notes);
  }
  return 0;
}
