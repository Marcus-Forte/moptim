#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

namespace moptim {

/// How a cost splits its elements between the work-items of one launch.
///  - `Chunked`: whole chunks of elements per work-item, each work-item owning its scratch and
///    accumulating partial sums (for CPUs; see ChunkedCost.hh).
///  - `WorkGroup`: one element per work-item, the work-group sums its elements' contributions together in
///    local memory (for GPUs and other accelerators; see detail/WorkGroupKernel.hh). A cost whose model
///    cannot be called through a const reference uses Chunked instead.
///  - `Auto`: `WorkGroup` on a device that is not a CPU, `Chunked` on a CPU.
enum class AssemblyStrategy { Auto, Chunked, WorkGroup };

/**
 * @brief The SYCL queue the costs launch on, and the knobs that decide how wide a launch is.
 *
 * - The queue is in-order, so a cost can enqueue its kernels back to back.
 * - `threads` bounds the number of work-groups of one launch (`0` = all compute units of the
 *   device). It only changes how the work is shared out, never the result: partial sums are
 *   combined in an order that does not depend on it (see ChunkedCost.hh).
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

  /// True if a cost evaluated on this context uses the WorkGroup kernel (when its model allows it):
  /// the strategy is `WorkGroup`, or `Auto` on a device that is not a CPU.
  bool usesWorkGroupKernel() const {
    return strategy_ == AssemblyStrategy::WorkGroup || (strategy_ == AssemblyStrategy::Auto && !is_cpu_);
  }

  /// Work-items per work-group of the WorkGroup kernel; 0 (the default) means 128, or the device's
  /// maximum if that is smaller. A cost may use fewer to fit its local memory.
  size_t workGroupSize() const { return work_group_size_; }
  void setWorkGroupSize(size_t size) { work_group_size_ = size; }
  /// workGroupSize() with 0 resolved and capped by the device's maximum work-group size.
  size_t resolvedWorkGroupSize() const {
    return std::max<size_t>(1, std::min(work_group_size_ != 0 ? work_group_size_ : size_t{128}, max_work_group_size_));
  }
  /// Local memory of a work-group, in bytes.
  size_t localMemoryBytes() const { return local_memory_bytes_; }

  /// Kernel launches issued through `detail::launch()` so far.
  uint64_t launches() const { return launches_.load(std::memory_order_relaxed); }

 private:
  friend struct LaunchAccess;

  sycl::queue queue_;
  size_t threads_;
  AssemblyStrategy strategy_;
  size_t compute_units_;
  size_t work_group_size_ = 0;
  size_t max_work_group_size_;
  size_t local_memory_bytes_;
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
 * Starts `kernel` (a callable taking `sycl::nd_item<1>`) on `work_groups` work-groups of one
 * work-item each and returns without waiting; complete() (or a later wait on the queue) finishes it.
 * The kernel is named `Name`: one that is instantiated with a type of the caller puts `CurrentTuId`
 * (detail/KernelName.hh) in its name, or same-named types of other translation units would share a
 * kernel. Counted as one launch. Does nothing for `work_groups == 0`. On native_cpu a launch of one
 * work-group runs inside this call, on the calling thread.
 */
template <class Name, class Kernel>
void submitNamed(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  LaunchAccess::count(context);
  sycl::ext::oneapi::experimental::nd_launch<Name>(context.queue(), sycl::nd_range<1>{work_groups, 1}, kernel);
}

/// Waits for everything submitted on the context's queue and rethrows asynchronous errors.
inline void complete(ExecutionContext& context) { context.queue().wait_and_throw(); }

/**
 * Starts a kernel on `groups` work-groups of `local_size` work-items with `local_values` elements of
 * work-group local memory of type `T` and one `unsigned int` of local memory more, and returns without
 * waiting. `kernel` is called as `kernel(sycl::nd_item<1>, T* local, unsigned int* local_flag)` and
 * is named `Name` (see submitNamed()). Counted as one launch. Does nothing for `groups == 0`.
 */
template <class Name, class T, class Kernel>
void submitWithLocalMemory(ExecutionContext& context, size_t groups, size_t local_size, size_t local_values,
                           Kernel kernel) {
  if (groups == 0) {
    return;
  }
  LaunchAccess::count(context);
  context.queue().submit([&](sycl::handler& handler) {
    sycl::local_accessor<T, 1> local(sycl::range<1>(std::max<size_t>(local_values, 1)), handler);
    sycl::local_accessor<unsigned int, 1> flag(sycl::range<1>(1), handler);
    handler.parallel_for<Name>(sycl::nd_range<1>(groups * local_size, local_size), [=](sycl::nd_item<1> item) {
      kernel(item, local.template get_multi_ptr<sycl::access::decorated::no>().get(),
             flag.template get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

/// submitNamed() followed by complete(). One launch costs about 10 to 25 microseconds on native_cpu.
template <class Name, class Kernel>
void launchNamed(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  submitNamed<Name>(context, work_groups, kernel);
  complete(context);
}

/// launchNamed() with the kernel named after its own type (for kernels of the library's own types).
template <class Kernel>
void launch(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  launchNamed<Kernel>(context, work_groups, kernel);
}

}  // namespace detail
}  // namespace moptim
