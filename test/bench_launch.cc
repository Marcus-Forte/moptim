// Micro-benchmark of SYCL launch latency and multi-core scaling on the default device.
//
//   bench_launch [section ...]     sections: latency pingpong scaling threads
//                                            phases workers alternatives (default: all)
//
// latency   round trip (submit + wait) of a kernel that writes one value to USM, for several ways
//           of submitting it, on an in-order and on an out-of-order queue.
// pingpong  reference: condition-variable round trip between two std::threads.
// scaling   nd_range<1>{W, 1} where every work-item runs the same ~20 ms dependent FMA chain;
//           speedup = W * T(1) / T(W), i.e. W when the work-groups run fully in parallel.
// phases    where a single-work-group launch spends its time: submit call vs wait.
// workers   trivial launches of W = 1, 2, 4, 11 work-groups (thread-pool wake-up cost).
// alternatives  other ways to cut the round trip: queue properties, sycl_ext_oneapi_graph replay,
//           kernel bundle, barrier, and completion by spinning on a USM flag instead of wait().
//           Run the whole program under SYCL_NATIVE_CPU_HOST_THREADS=1 / =2 to see the pool size.
// threads   which threads of this process accumulate CPU time during one single-work-item launch
//           (reads /proc/self/task/*/stat, Linux only).
//
// Not run by ctest. Timings on a loaded machine are noisy.
// sycl::khr::launch_grouped is still flagged unfinished in this DPC++ release.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace exp_sycl = sycl::ext::oneapi::experimental;

double microsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

struct Stats {
  double min, median, p99;
};

Stats summarize(std::vector<double> us) {
  std::sort(us.begin(), us.end());
  return {us.front(), us[us.size() / 2], us[static_cast<size_t>(0.99 * static_cast<double>(us.size() - 1))]};
}

void printLatency(const char* name, const std::vector<double>& us) {
  const Stats s = summarize(us);
  std::printf("  %-34s min %8.2f  median %8.2f  p99 %9.2f us\n", name, s.min, s.median, s.p99);
  std::fflush(stdout);
}

constexpr int kWarmup = 1000;
constexpr int kCalls = 10000;

/// Times `kCalls` calls of `roundTrip` (after warm-up), in microseconds each.
template <class RoundTrip>
std::vector<double> measure(RoundTrip&& roundTrip) {
  for (int i = 0; i < kWarmup; ++i) roundTrip(i);
  std::vector<double> us;
  us.reserve(kCalls);
  for (int i = 0; i < kCalls; ++i) {
    const auto start = Clock::now();
    roundTrip(i);
    us.push_back(microsSince(start));
  }
  return us;
}

void latencySection(const sycl::device& device) {
  std::printf("\n== latency: submit + wait of a kernel writing one value to USM (%d calls) ==\n", kCalls);
  for (const bool in_order : {true, false}) {
    sycl::queue q = in_order ? sycl::queue{device, {sycl::property::queue::in_order()}} : sycl::queue{device};
    std::printf(" %s queue\n", in_order ? "in-order" : "out-of-order");
    int* out = sycl::malloc_shared<int>(1, q);
    *out = 0;

    printLatency("khr::launch_grouped {1,1}", measure([&](int i) {
                   sycl::khr::launch_grouped(q, sycl::range<1>(1), sycl::range<1>(1),
                                             [=](sycl::nd_item<1>) { *out = i; });
                   q.wait();
                 }));
    printLatency("nd_launch", measure([&](int i) {
                   exp_sycl::nd_launch(q, sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) { *out = i; });
                   q.wait();
                 }));
    printLatency("submit(parallel_for nd_range)", measure([&](int i) {
                   q.submit([&](sycl::handler& h) {
                      h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) { *out = i; });
                    }).wait();
                 }));
    printLatency("single_task", measure([&](int i) { q.single_task([=] { *out = i; }).wait(); }));
    if (*out != kCalls - 1) std::printf("  WARNING: last written value %d\n", *out);
    sycl::free(out, q);
  }
}

void pingPongSection() {
  std::printf("\n== pingpong: condition-variable round trip between two threads (%d calls) ==\n", kCalls);
  std::mutex mutex;
  std::condition_variable cv;
  int request = 0;  // written by the main thread
  int reply = 0;    // written by the worker
  bool stop = false;

  std::thread worker([&] {
    std::unique_lock lock(mutex);
    int seen = 0;
    while (true) {
      cv.wait(lock, [&] { return stop || request != seen; });
      if (stop) return;
      seen = request;
      reply = seen;
      cv.notify_all();
    }
  });

  printLatency("cv ping-pong", measure([&](int i) {
                 std::unique_lock lock(mutex);
                 request = i + 1;
                 cv.notify_all();
                 cv.wait(lock, [&] { return reply == request; });
               }));
  {
    std::lock_guard lock(mutex);
    stop = true;
  }
  cv.notify_all();
  worker.join();
}

/// Each work-item runs `iterations` steps of a dependent FMA chain and stores the result.
/// `a` and `b` come from USM so the compiler cannot fold the chain.
struct Chain {
  const double* params;  // {a, b}
  double* out;
  long iterations;
};

void launchChain(sycl::queue& q, const Chain& c, size_t workers) {
  sycl::khr::launch_grouped(q, sycl::range<1>(workers), sycl::range<1>(1), [=](sycl::nd_item<1> item) {
    const double a = c.params[0];
    const double b = c.params[1];
    double x = 1.0 + 1e-3 * static_cast<double>(item.get_group(0));
    for (long i = 0; i < c.iterations; ++i) x = x * a + b;
    c.out[item.get_group(0)] = x;
  });
  q.wait();
}

/// Iteration count for which one work-item runs about `target_ms`.
long calibrate(sycl::queue& q, double* params, double* out, double target_ms) {
  long iterations = 20'000'000;
  for (int attempt = 0; attempt < 3; ++attempt) {
    launchChain(q, {params, out, iterations}, 1);  // warm-up (thread pool start-up, page faults)
    const auto start = Clock::now();
    launchChain(q, {params, out, iterations}, 1);
    const double ms = microsSince(start) / 1e3;
    iterations = static_cast<long>(static_cast<double>(iterations) * target_ms / ms);
  }
  return iterations;
}

void scalingSection(sycl::queue& q, size_t max_units) {
  std::printf("\n== scaling: nd_range<1>{W, 1}, every work-item runs the same ~20 ms FMA chain ==\n");
  double* params = sycl::malloc_shared<double>(2, q);
  double* out = sycl::malloc_shared<double>(max_units, q);
  params[0] = 0.99999999;
  params[1] = 1e-9;
  const long iterations = calibrate(q, params, out, 20.0);
  std::printf("  iterations per work-item: %ld\n  %3s %12s %10s %10s\n", iterations, "W", "time ms", "speedup", "efficiency");

  double t1 = 0.0;
  for (size_t workers = 1; workers <= max_units; ++workers) {
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      const auto start = Clock::now();
      launchChain(q, {params, out, iterations}, workers);
      best = std::min(best, microsSince(start) / 1e3);
    }
    if (workers == 1) t1 = best;
    const double speedup = static_cast<double>(workers) * t1 / best;
    std::printf("  %3zu %12.3f %10.2f %9.0f%%\n", workers, best, speedup, 100.0 * speedup / static_cast<double>(workers));
    std::fflush(stdout);
  }
  sycl::free(params, q);
  sycl::free(out, q);
}

struct ThreadTimes {
  std::vector<std::pair<std::string, long>> ticks;  // (tid + name, utime + stime)
};

ThreadTimes readThreadTimes() {
  ThreadTimes times;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
    std::ifstream stat(entry.path() / "stat");
    std::stringstream buffer;
    buffer << stat.rdbuf();
    const std::string line = buffer.str();
    const size_t open = line.find('(');
    const size_t close = line.rfind(')');
    if (open == std::string::npos || close == std::string::npos) continue;  // the thread exited
    std::istringstream rest(line.substr(close + 2));
    // Fields after the name start at field 3 (state); utime and stime are fields 14 and 15.
    std::string field;
    long utime = 0, stime = 0;
    for (int i = 3; i <= 15; ++i) {
      rest >> field;
      if (i == 14) utime = std::stol(field);
      if (i == 15) stime = std::stol(field);
    }
    times.ticks.emplace_back(entry.path().filename().string() + " " + line.substr(open, close - open + 1), utime + stime);
  }
  return times;
}

void threadsSection(sycl::queue& q) {
  std::printf("\n== threads: CPU time per thread during a single-work-item launch (~300 ms of work) ==\n");
  double* params = sycl::malloc_shared<double>(2, q);
  double* out = sycl::malloc_shared<double>(1, q);
  params[0] = 0.99999999;
  params[1] = 1e-9;
  const long iterations = calibrate(q, params, out, 20.0) * 15;

  const ThreadTimes before = readThreadTimes();
  const auto start = Clock::now();
  launchChain(q, {params, out, iterations}, 1);
  const double ms = microsSince(start) / 1e3;
  const ThreadTimes after = readThreadTimes();

  const long tick_hz = sysconf(_SC_CLK_TCK);
  int busy = 0;
  for (const auto& [name, ticks] : after.ticks) {
    long was = 0;
    for (const auto& [old_name, old_ticks] : before.ticks) {
      if (old_name == name) was = old_ticks;
    }
    if (ticks - was > 0) {
      ++busy;
      std::printf("  %-28s +%ld ticks (%.0f ms)\n", name.c_str(), ticks - was,
                  1e3 * static_cast<double>(ticks - was) / static_cast<double>(tick_hz));
    }
  }
  std::printf("  wall time %.1f ms; %d of %zu tasks accumulated CPU time (before: %zu tasks)\n", ms, busy,
              after.ticks.size(), before.ticks.size());
  sycl::free(params, q);
  sycl::free(out, q);
}


// ---- Launch latency experiments ---------------------------------------------------------------

class BundleKernel;  // named kernel, so that it can be looked up in a kernel bundle

void printSplit(const char* name, const std::vector<double>& submit, const std::vector<double>& wait,
                const std::vector<double>& total) {
  const Stats a = summarize(submit), b = summarize(wait), c = summarize(total);
  std::printf("  %-30s submit %6.2f/%6.2f/%6.2f  wait %6.2f/%6.2f/%6.2f  total %6.2f/%6.2f/%6.2f\n", name, a.min,
              a.median, a.p99, b.min, b.median, b.p99, c.min, c.median, c.p99);
  std::fflush(stdout);
}

/// Times the submit call and the wait of `kCalls` round trips separately (min/median/p99, us).
template <class Submit, class Wait>
void measureSplit(const char* name, Submit&& submit, Wait&& wait) {
  std::vector<double> s, w, t;
  for (int i = 0; i < kWarmup + kCalls; ++i) {
    const auto start = Clock::now();
    submit(i);
    const auto mid = Clock::now();
    wait();
    const auto end = Clock::now();
    if (i >= kWarmup) {
      s.push_back(std::chrono::duration<double, std::micro>(mid - start).count());
      w.push_back(std::chrono::duration<double, std::micro>(end - mid).count());
      t.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
  }
  printSplit(name, s, w, t);
}

void phasesSection(const sycl::device& device) {
  std::printf("\n== phases: submit call vs wait, single work-group (us, min/median/p99) ==\n");
  sycl::queue q{device, {sycl::property::queue::in_order()}};
  int* out = sycl::malloc_shared<int>(1, q);
  measureSplit("launch_grouped, q.wait()", [&](int i) {
    sycl::khr::launch_grouped(q, sycl::range<1>(1), sycl::range<1>(1), [=](sycl::nd_item<1>) { *out = i; });
  }, [&] { q.wait(); });
  sycl::event ev;
  measureSplit("q.submit, event.wait()", [&](int i) {
    ev = q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) { *out = i; }); });
  }, [&] { ev.wait(); });
  measureSplit("single_task, event.wait()", [&](int i) { ev = q.single_task([=] { *out = i; }); }, [&] { ev.wait(); });
  // Lower bounds: a runtime round trip with no kernel at all.
  measureSplit("empty q.wait()", [](int) {}, [&] { q.wait(); });
  measureSplit("submit_barrier, event.wait()", [&](int) { ev = q.ext_oneapi_submit_barrier(); }, [&] { ev.wait(); });
  sycl::free(out, q);
}

void workersSection(const sycl::device& device) {
  std::printf("\n== workers: trivial nd_launch of W work-groups, submit + wait (us, min/median/p99) ==\n");
  sycl::queue q{device, {sycl::property::queue::in_order()}};
  int* out = sycl::malloc_shared<int>(64, q);
  for (const size_t workers : {size_t{1}, size_t{2}, size_t{4}, size_t{11}}) {
    char name[32];
    std::snprintf(name, sizeof(name), "W=%zu", workers);
    printLatency(name, measure([&](int i) {
                   exp_sycl::nd_launch(q, sycl::nd_range<1>(workers, 1),
                                       [=](sycl::nd_item<1> item) { out[item.get_group(0)] = i; });
                   q.wait();
                 }));
  }
  sycl::free(out, q);
}

void alternativesSection(const sycl::device& device) {
  std::printf("\n== alternatives: single work-group round trip (us, min/median/p99) ==\n");
  const auto trivial = [](sycl::queue& q, int* out) {
    return [&q, out](int i) {
      sycl::khr::launch_grouped(q, sycl::range<1>(1), sycl::range<1>(1), [=](sycl::nd_item<1>) { *out = i; });
      q.wait();
    };
  };
  {
    sycl::queue q{device, {sycl::property::queue::in_order()}};
    int* out = sycl::malloc_shared<int>(1, q);
    printLatency("baseline (in-order)", measure(trivial(q, out)));
    sycl::free(out, q);
  }
  {
    // Deprecated in favour of the enqueue functions, which is what the other rows already use.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    sycl::queue q{device, {sycl::property::queue::in_order(), sycl::ext::oneapi::property::queue::discard_events()}};
#pragma clang diagnostic pop
    int* out = sycl::malloc_shared<int>(1, q);
    printLatency("discard_events property", measure(trivial(q, out)));
    sycl::free(out, q);
  }
  {
    sycl::queue q{device, {sycl::property::queue::in_order(), sycl::ext::intel::property::queue::immediate_command_list()}};
    int* out = sycl::malloc_shared<int>(1, q);
    printLatency("immediate_command_list (L0 only)", measure(trivial(q, out)));
    sycl::free(out, q);
  }
  {
    sycl::queue q{device, {sycl::property::queue::in_order(), sycl::ext::oneapi::property::queue::priority_high()}};
    int* out = sycl::malloc_shared<int>(1, q);
    printLatency("priority_high property", measure(trivial(q, out)));
    sycl::free(out, q);
  }
  {
    // Kernel looked up in a pre-built executable kernel bundle.
    sycl::queue q{device, {sycl::property::queue::in_order()}};
    int* out = sycl::malloc_shared<int>(1, q);
    try {
      auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {sycl::get_kernel_id<BundleKernel>()});
      printLatency("kernel bundle (use_kernel_bundle)", measure([&](int i) {
                     q.submit([&](sycl::handler& h) {
                        h.use_kernel_bundle(bundle);
                        h.parallel_for<BundleKernel>(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) { *out = i; });
                      }).wait();
                   }));
    } catch (const sycl::exception& e) {
      std::printf("  kernel bundle: failed: %s\n", e.what());
    }
    sycl::free(out, q);
  }
  {
    // Graph recorded once, replayed per call.
    sycl::queue q{device, {sycl::property::queue::in_order()}};
    int* out = sycl::malloc_shared<int>(1, q);
    namespace exg = sycl::ext::oneapi::experimental;
    if (!device.has(sycl::aspect::ext_oneapi_graph)) {
      std::printf("  graph: device lacks aspect ext_oneapi_graph\n");
    } else {
      try {
        exg::command_graph graph{q.get_context(), device};
        graph.begin_recording(q);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) { *out = 42; }); });
        graph.end_recording(q);
        auto exec = graph.finalize();
        *out = 0;
        printLatency("graph replay + wait", measure([&](int) { q.ext_oneapi_graph(exec).wait(); }));
        if (*out != 42) std::printf("  WARNING: graph kernel wrote %d\n", *out);
      } catch (const sycl::exception& e) {
        std::printf("  graph: failed: %s\n", e.what());
      }
    }
    sycl::free(out, q);
  }
  {
    // Completion through a flag the kernel writes; no q.wait() per call. The queue is drained every
    // 1000 calls and at the end. Only valid where the device and host share coherent memory.
    sycl::queue q{device, {sycl::property::queue::in_order()}};
    unsigned long* flag = sycl::malloc_shared<unsigned long>(1, q);
    *flag = 0;
    unsigned long seq = 0;
    printLatency("USM flag spin (no wait)", measure([&](int) {
                   const unsigned long expected = ++seq;
                   sycl::khr::launch_grouped(q, sycl::range<1>(1), sycl::range<1>(1), [=](sycl::nd_item<1>) {
                     sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed, sycl::memory_scope::system> f(*flag);
                     f.store(expected, sycl::memory_order::release);
                   });
                   sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed, sycl::memory_scope::system> f(*flag);
                   while (f.load(sycl::memory_order::acquire) != expected) {
                   }
                   if (expected % 1000 == 0) q.wait();
                 }));
    q.wait();
    sycl::free(flag, q);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> sections;
  for (int i = 1; i < argc; ++i) sections.emplace_back(argv[i]);
  const auto wants = [&](const char* name) {
    return sections.empty() || std::find(sections.begin(), sections.end(), name) != sections.end();
  };

  sycl::queue q;
  const sycl::device device = q.get_device();
  const size_t units = device.get_info<sycl::info::device::max_compute_units>();
  std::printf("device: %s\nmax_compute_units: %zu\n", device.get_info<sycl::info::device::name>().c_str(), units);
  if (const char* pool = std::getenv("SYCL_NATIVE_CPU_HOST_THREADS")) {
    std::printf("SYCL_NATIVE_CPU_HOST_THREADS=%s\n", pool);
  }

  if (wants("latency")) latencySection(device);
  if (wants("pingpong")) pingPongSection();
  if (wants("scaling")) scalingSection(q, units);
  if (wants("threads")) threadsSection(q);
  if (wants("phases")) phasesSection(device);
  if (wants("workers")) workersSection(device);
  if (wants("alternatives")) alternativesSection(device);
  return 0;
}
