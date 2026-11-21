#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

namespace moptim {

/// How a cost splits its elements between the work-items of one launch (used from Phase 2 on; the
/// costs do not read it yet). `Auto` lets the cost choose from the device and the problem size.
enum class AssemblyStrategy { Auto, Chunked, WorkGroup };

/**
 * @brief The SYCL queue the costs launch on, and the knobs that decide how wide a launch is.
 *
 * - The queue is in-order, so a cost can enqueue its kernels back to back.
 * - `threads` bounds the number of work-groups of one launch (`0` = all compute units of the
 *   device). It only changes how the work is shared out, never the result: partial sums are
 *   combined in an order that does not depend on it.
 * - `launches()` counts the kernel launches issued through `detail::launch()`, so a test or a
 *   benchmark can check how many launches an operation takes.
 *
 * Setters are not synchronized with launches: configure the context before using it from several
 * threads. Not copyable; share it through `std::shared_ptr`.
 */
class ExecutionContext {
 public:
  /// `sycl::default_selector_v`, in-order queue.
  ExecutionContext();
  /// Uses `queue`, which must be in-order.
  explicit ExecutionContext(sycl::queue queue, size_t threads = 0, AssemblyStrategy strategy = AssemblyStrategy::Auto);

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  sycl::queue& queue() { return queue_; }

  /// 0 means all compute units.
  size_t threads() const { return threads_; }
  void setThreads(size_t threads) { threads_ = threads; }

  /// Number of work-groups to launch for `partitions` independent pieces of work:
  /// `min(threads or compute units, partitions)`, at least 1 when `partitions > 0`, 0 otherwise.
  size_t launchWidth(size_t partitions) const;

  /// The device can dereference host allocations (`usm_system_allocations`), so costs can read the
  /// caller's arrays without a copy.
  bool hostAccessible() const { return host_accessible_; }
  bool isCpu() const { return is_cpu_; }

  AssemblyStrategy strategy() const { return strategy_; }
  void setStrategy(AssemblyStrategy strategy) { strategy_ = strategy; }

  /// Kernel launches issued through `detail::launch()` so far.
  uint64_t launches() const { return launches_.load(std::memory_order_relaxed); }

 private:
  friend struct LaunchAccess;

  sycl::queue queue_;
  size_t threads_;
  AssemblyStrategy strategy_;
  size_t compute_units_;
  bool host_accessible_;
  bool is_cpu_;
  std::atomic<uint64_t> launches_{0};
};

/// The process-wide context used by costs that were not given one. Created on first use.
std::shared_ptr<ExecutionContext> defaultContext();

/// Replaces the process-wide context. Costs created earlier keep the one they captured. `nullptr`
/// discards the current one, so that the next defaultContext() creates a fresh default.
void setDefaultContext(std::shared_ptr<ExecutionContext> context);

/// Counter access for detail::launch().
struct LaunchAccess {
  static void count(ExecutionContext& context) { context.launches_.fetch_add(1, std::memory_order_relaxed); }
};

namespace detail {

/**
 * Runs `kernel` (a callable taking `sycl::nd_item<1>`) on `work_groups` work-groups of one
 * work-item each, and waits for it. Rethrows asynchronous errors. One launch costs about 10 to 25
 * microseconds on native_cpu (Phase 0 measurements), and a launch of one work-group runs on the
 * calling thread. Does nothing for `work_groups == 0`.
 */
template <class Kernel>
void launch(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  LaunchAccess::count(context);
  sycl::ext::oneapi::experimental::nd_launch(context.queue(), sycl::nd_range<1>{work_groups, 1}, kernel);
  context.queue().wait_and_throw();
}

}  // namespace detail
}  // namespace moptim
