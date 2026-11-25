#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace moptim {

/// What an evaluation computes: a linear system, or the cost alone (about 15 times less work per partition,
/// so it takes more partitions to pay for using several threads).
enum class LaunchWork { LinearSystem, CostOnly };

namespace detail {
/// The oneTBB arena of a context; defined in src/ExecutionContext.cc, so no public header names TBB.
class HostExecutor;
}  // namespace detail

/**
 * @brief How many threads the costs may use, and where they run.
 *
 * - An evaluation of a cost splits its work into partitions (see ChunkedCost.hh) and hands them to one
 *   worker per thread; a worker claims whole partitions. A small cost, or `threads == 1`, is evaluated by
 *   one worker inline on the calling thread; a large one by several workers on a oneTBB `task_arena` owned
 *   by the context, with the calling thread taking part. The evaluation returns when all partitions are
 *   done. An exception thrown on a worker reaches the caller.
 * - `threads` is the arena's maximum concurrency, which is also the most workers of one evaluation
 *   (`0` = all hardware threads, `tbb::info::default_concurrency()`). It only changes how the work is
 *   shared out, never the result: partitions and the order of the reduction depend on the problem alone
 *   (see ChunkedCost.hh), so every thread count gives the same bits.
 * - Without oneTBB (`parallelAvailable()` is false) every evaluation runs inline, whatever `threads` says.
 * - `launches()` counts the evaluations dispatched through the context (a cost evaluated on several
 *   threads still counts as one), so a test or a benchmark can check how many an operation takes.
 *
 * The arena is created in the constructor and re-created by setThreads() (never lazily, so there is no
 * race on first use). Setters are not synchronized with evaluations: configure the context before using it
 * from several threads. Not copyable; share it through `std::shared_ptr`.
 */
class ExecutionContext {
 public:
  /// All hardware threads (`threads() == 0`).
  ExecutionContext();
  /// At most @p threads workers per evaluation (0 = all hardware threads).
  explicit ExecutionContext(size_t threads);
  ~ExecutionContext();

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  /// Whether this moptim was built with oneTBB (MOPTIM_WITH_TBB and TBB found): large costs can then be
  /// evaluated on several threads.
  static bool parallelAvailable();

  /// The requested number of threads; 0 means all hardware threads.
  size_t threads() const { return threads_; }
  /// Changes the thread count and re-creates the arena. Not safe during an evaluation.
  void setThreads(size_t threads);
  /// threads() with 0 resolved to the number of hardware threads; 1 without oneTBB.
  size_t resolvedThreads() const;

  /// Number of workers (threads) to use for `partitions` independent pieces of work: 0 for none, 1 (inline
  /// on the calling thread) below a minimum number of partitions, otherwise
  /// `min(resolvedThreads(), ceil(partitions / 2))`. Waking the arena's workers costs several
  /// microseconds, which a small cost does not earn back: the minimum is
  /// kMinPartitionsToGoWide (6) for a linear system and kMinPartitionsToGoWideCostOnly (12) for a cost-only
  /// evaluation, measured where threads 0 stops being slower than threads 1. Always 1 without oneTBB. The
  /// width never changes a result.
  size_t launchWidth(size_t partitions, LaunchWork work = LaunchWork::LinearSystem) const;

  /// Partitions an evaluation needs before it uses several threads (see launchWidth()).
  static constexpr size_t kMinPartitionsToGoWide = 6;
  static constexpr size_t kMinPartitionsToGoWideCostOnly = 12;

  /// Evaluations dispatched through this context so far (one per evaluation, whatever its thread count).
  uint64_t launches() const { return launches_.load(std::memory_order_relaxed); }

  /// Runs `function(data, g)` for every task g in [0, groups) and returns when all are done. One task
  /// runs inline on the calling thread, as does everything without oneTBB (in order); otherwise the tasks
  /// run concurrently on the arena. Rethrows an exception a task threw (the first one; the tasks not yet
  /// started are skipped).
  void runOnHost(size_t groups, void (*function)(void*, size_t), void* data);

 private:
  friend struct LaunchAccess;

  void createExecutor();

  size_t threads_;
  std::atomic<uint64_t> launches_{0};
  std::unique_ptr<detail::HostExecutor> executor_;  ///< Null without oneTBB.
};

/// The process-wide context used by costs that were not given one: all hardware threads. Created on
/// first use.
std::shared_ptr<ExecutionContext> defaultContext();

/// Replaces the process-wide context. Costs created earlier keep the one they captured. `nullptr`
/// discards the current one, so that the next defaultContext() creates a fresh default.
void setDefaultContext(std::shared_ptr<ExecutionContext> context);

/// Counter access for detail::submit().
struct LaunchAccess {
  static void count(ExecutionContext& context) { context.launches_.fetch_add(1, std::memory_order_relaxed); }
};

namespace detail {

/// Runs `kernel(task)` for every task of an evaluation (type-erased for runOnHost()).
template <class Kernel>
void runKernelOnHost(ExecutionContext& context, size_t groups, Kernel& kernel) {
  context.runOnHost(
      groups, [](void* data, size_t group) { (*static_cast<Kernel*>(data))(group); }, &kernel);
}

/**
 * Runs `kernel` (a callable taking the task index, `size_t`) for `work_groups` tasks, one per worker, and
 * returns when they are all done. Counted as one evaluation by `launches()`. Does nothing for
 * `work_groups == 0`. The tasks of one evaluation may run concurrently, so `kernel` must be safe to call
 * from several threads.
 */
template <class Kernel>
void submit(ExecutionContext& context, size_t work_groups, Kernel kernel) {
  if (work_groups == 0) {
    return;
  }
  LaunchAccess::count(context);
  runKernelOnHost(context, work_groups, kernel);
}

/// Runs `kernel()` on the calling thread as one evaluation (counted as one): for work that must not be
/// split across threads, such as a walk that calls a model's `setState()`.
template <class Kernel>
void runOnCallingThread(ExecutionContext& context, Kernel&& kernel) {
  LaunchAccess::count(context);
  kernel();
}

}  // namespace detail
}  // namespace moptim
