#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#if MOPTIM_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

namespace moptim {

/// How a cost splits its elements between the work-items of one launch.
///  - `Chunked`: whole chunks of elements per work-item, each work-item owning its scratch and
///    accumulating partial sums (for CPUs; see ChunkedCost.hh).
///  - `WorkGroup`: one element per work-item, the work-group sums its elements' contributions together in
///    local memory (for GPUs and other accelerators; see detail/WorkGroupKernel.hh). A cost whose model
///    cannot be called through a const reference uses Chunked instead, and so does the host backend.
///  - `Auto`: `WorkGroup` on a SYCL device that is not a CPU, `Chunked` otherwise.
enum class AssemblyStrategy { Auto, Chunked, WorkGroup };

/**
 * @brief Where a context runs the costs' work.
 *  - `Host`: sequentially on the calling thread (the reference path; no host threads). Always available.
 *  - `Sycl`: in SYCL kernels on a device. Only in a build with MOPTIM_WITH_SYCL.
 *
 * Both run the same work-item functions (detail/ChunkKernel.hh), so a cost, a FusedCost or a
 * DeviceLevenbergMarquardt behaves the same on either; the arithmetic of a SYCL device pass may differ
 * from the host's in the last bits (e.g. scalar Eigen in the device pass).
 */
enum class Backend { Host, Sycl };

/// Whether this moptim was built with SYCL (MOPTIM_WITH_SYCL).
constexpr bool syclAvailable() {
#if MOPTIM_HAS_SYCL
  return true;
#else
  return false;
#endif
}

/// The backend of default-constructed contexts: `MOPTIM_BACKEND` (`host` or `sycl`) if set, otherwise
/// Sycl in a build with SYCL and Host without. A `sycl` request in a build without SYCL gives Host.
Backend defaultBackend();

/// "host" or "sycl".
const char* backendName(Backend backend);

/**
 * @brief The backend the costs run on, and the knobs that decide how wide a launch is.
 *
 * - A SYCL context owns an in-order queue, so a cost can enqueue its kernels back to back.
 * - A host context runs every work-group of a launch in order on the calling thread and returns when it
 *   is done. moptim has no host threads: parallelism comes only from the SYCL backend.
 * - `threads` bounds the number of work-groups of one launch on a SYCL device (`0` = all compute units
 *   of the device). It is ignored on the host, which always runs one work-group. It only changes how the work is shared out, never the
 *   result: partial sums are combined in an order that does not depend on it (see ChunkedCost.hh).
 * - `launches()` counts the launches issued through `detail::submitNamed()`, so a test or a
 *   benchmark can check how many launches an operation takes, on either backend.
 *
 * Setters are not synchronized with launches: configure the context before using it from several
 * threads. Not copyable; share it
 * through `std::shared_ptr`.
 */
class ExecutionContext {
 public:
  /// defaultBackend(), all compute units.
  ExecutionContext();
  /// @p backend (Sycl: `sycl::default_selector_v`, in-order queue). Throws std::invalid_argument for
  /// Sycl in a build without SYCL.
  explicit ExecutionContext(Backend backend, size_t threads = 0, AssemblyStrategy strategy = AssemblyStrategy::Auto);
#if MOPTIM_HAS_SYCL
  /// Sycl backend on `queue`, which must be in-order.
  explicit ExecutionContext(sycl::queue queue, size_t threads = 0, AssemblyStrategy strategy = AssemblyStrategy::Auto);
  /// The SYCL queue (Sycl backend only).
  sycl::queue& queue() { return *queue_; }
#endif
  ~ExecutionContext();

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  Backend backend() const { return backend_; }

  /// Work-groups per launch on a SYCL device (0 means all compute units); ignored on the host.
  size_t threads() const { return threads_; }
  void setThreads(size_t threads) { threads_ = threads; }

  /// Number of work-groups to launch for `partitions` independent pieces of work, at least 1 when
  /// `partitions > 0`, 0 otherwise: `min(threads or compute units, partitions)` on a SYCL device; always 1
  /// on the host.
  size_t launchWidth(size_t partitions) const;

  /// The device can dereference host allocations (`usm_system_allocations`; always on the host), so
  /// costs can read the caller's arrays without a copy.
  bool hostAccessible() const { return host_accessible_; }
  bool isCpu() const { return is_cpu_; }

  AssemblyStrategy strategy() const { return strategy_; }
  void setStrategy(AssemblyStrategy strategy) { strategy_ = strategy; }

  /// True if a cost evaluated on this context uses the WorkGroup kernel (when its model allows it): a
  /// SYCL context whose strategy is `WorkGroup`, or `Auto` on a device that is not a CPU.
  bool usesWorkGroupKernel() const {
    return backend_ == Backend::Sycl &&
           (strategy_ == AssemblyStrategy::WorkGroup || (strategy_ == AssemblyStrategy::Auto && !is_cpu_));
  }

  /// Work-items per work-group of the WorkGroup kernel; 0 (the default) means 128, or the device's
  /// maximum if that is smaller. A cost may use fewer to fit its local memory.
  size_t workGroupSize() const { return work_group_size_; }
  void setWorkGroupSize(size_t size) { work_group_size_ = size; }
  /// workGroupSize() with 0 resolved and capped by the device's maximum work-group size.
  size_t resolvedWorkGroupSize() const {
    return std::max<size_t>(1, std::min(work_group_size_ != 0 ? work_group_size_ : size_t{128}, max_work_group_size_));
  }
  /// Local memory of a work-group, in bytes (0 on the host).
  size_t localMemoryBytes() const { return local_memory_bytes_; }

  /// Launches issued through `detail::submitNamed()` so far.
  uint64_t launches() const { return launches_.load(std::memory_order_relaxed); }

  // ─── Memory ───────────────────────────────────────────────────────────────────────────────────
  /// `bytes` (at least one) of memory both the host and the kernels can access, aligned to `alignment`
  /// (a power of two): shared USM on a SYCL context, plain memory on the host. Throws std::bad_alloc.
  void* allocateShared(size_t bytes, size_t alignment);
  /// Memory the kernels can access, possibly not the host (device USM; plain memory on the host).
  void* allocateDevice(size_t bytes, size_t alignment);
  /// Frees what allocateShared() / allocateDevice() returned (null is ignored).
  void deallocate(void* ptr) noexcept;
  /// Copies `bytes` from host memory into an allocation of this context, and waits.
  void copyToDevice(void* destination, const void* source, size_t bytes);

  // ─── Host launches ────────────────────────────────────────────────────────────────────────────
  /// Host backend: runs `function(data, g)` for every work-group g in [0, groups) in order on the
  /// calling thread. An exception of a work-group propagates and ends the launch.
  void runOnHost(size_t groups, void (*function)(void*, size_t), void* data);

 private:
  friend struct LaunchAccess;

  size_t resolvedThreads() const { return threads_ != 0 ? threads_ : compute_units_; }

  Backend backend_;
#if MOPTIM_HAS_SYCL
  void initSycl(sycl::queue queue);
  std::optional<sycl::queue> queue_;  ///< Sycl backend only.
#endif
  size_t threads_;
  AssemblyStrategy strategy_;
  size_t compute_units_ = 1;
  size_t work_group_size_ = 0;
  size_t max_work_group_size_ = 1;
  size_t local_memory_bytes_ = 0;
  bool host_accessible_ = true;
  bool is_cpu_ = true;
  std::atomic<uint64_t> launches_{0};
};

/// The process-wide context used by costs that were not given one, on defaultBackend(). Created on
/// first use.
std::shared_ptr<ExecutionContext> defaultContext();

/// Replaces the process-wide context. Costs created earlier keep the one they captured. `nullptr`
/// discards the current one, so that the next defaultContext() creates a fresh default.
void setDefaultContext(std::shared_ptr<ExecutionContext> context);

/// Counter access for detail::submitNamed().
struct LaunchAccess {
  static void count(ExecutionContext& context) { context.launches_.fetch_add(1, std::memory_order_relaxed); }
};

namespace detail {

/// Runs `kernel(group)` for every work-group of a launch on the host (type-erased for runOnHost()).
template <class Kernel>
void runKernelOnHost(ExecutionContext& context, size_t groups, Kernel& kernel) {
  context.runOnHost(
      groups, [](void* data, size_t group) { (*static_cast<Kernel*>(data))(group); }, &kernel);
}

/**
 * Starts `kernel` (a callable taking the work-group index, `size_t`) on `work_groups` work-groups of one
 * work-item each. On a SYCL context it returns without waiting (complete(), or a later wait on the
 * queue, finishes it); the kernel is named `Name`, and one instantiated with a type of the caller puts
 * `CurrentTuId` (detail/KernelName.hh) in its name, or same-named types of other translation units
 * would share a kernel. On a host context it runs the kernel and returns when it is done. Counted as one
 * launch. Does nothing for `work_groups == 0`. On native_cpu a launch of one work-group runs inside this
 * call, on the calling thread.
 */
template <class Name, class Kernel>
void submitNamed(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  LaunchAccess::count(context);
#if MOPTIM_HAS_SYCL
  if (context.backend() == Backend::Sycl) {
    const auto body = [kernel](sycl::nd_item<1> item) { kernel(item.get_group_linear_id()); };
    if constexpr (std::is_void_v<Name>) {
      // An unnamed kernel (detail::launch()): the wrapper's own type names it.
      sycl::ext::oneapi::experimental::nd_launch(context.queue(), sycl::nd_range<1>{work_groups, 1}, body);
    } else {
      sycl::ext::oneapi::experimental::nd_launch<Name>(context.queue(), sycl::nd_range<1>{work_groups, 1}, body);
    }
    return;
  }
#endif
  runKernelOnHost(context, work_groups, kernel);
}

/// Runs `kernel()` on the calling thread as one launch of a host context (counted as one): for a
/// launch whose body is host code, such as a walk that calls a model's host-only `setState()`.
template <class Kernel>
void runOnCallingThread(ExecutionContext& context, Kernel&& kernel) {
  LaunchAccess::count(context);
  kernel();
}

/// Waits for everything submitted on the context and rethrows asynchronous errors (nothing to wait for
/// on the host: its launches are done when they return).
inline void complete(ExecutionContext& context) {
#if MOPTIM_HAS_SYCL
  if (context.backend() == Backend::Sycl) {
    context.queue().wait_and_throw();
  }
#else
  (void)context;
#endif
}

#if MOPTIM_HAS_SYCL
/**
 * Starts a kernel on `groups` work-groups of `local_size` work-items with `local_values` elements of
 * work-group local memory of type `T` and one `unsigned int` of local memory more, and returns without
 * waiting. `kernel` is called as `kernel(sycl::nd_item<1>, T* local, unsigned int* local_flag)` and
 * is named `Name` (see submitNamed()). Counted as one launch. Does nothing for `groups == 0`. SYCL
 * contexts only (the WorkGroup kernel).
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
#endif

/// submitNamed() followed by complete(). One SYCL launch costs about 10 to 25 microseconds on
/// native_cpu; a host launch of one work-group is a function call.
template <class Name, class Kernel>
void launchNamed(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  submitNamed<Name>(context, work_groups, kernel);
  complete(context);
}

/// launchNamed() with an unnamed kernel (for kernels of the library's own types and tests).
template <class Kernel>
void launch(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  launchNamed<void>(context, work_groups, kernel);
}

}  // namespace detail
}  // namespace moptim
