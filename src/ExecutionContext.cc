#include "moptim/ExecutionContext.hh"

#include <algorithm>
#include <mutex>
#include <utility>

#if MOPTIM_HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>
#endif

namespace moptim {

namespace detail {

#if MOPTIM_HAS_TBB
/// The oneTBB arena of a context: its maximum concurrency is the context's resolved thread count, and
/// the thread that launches takes one of the slots. Created (and initialized) with the context.
class HostExecutor {
 public:
  explicit HostExecutor(size_t threads) : arena_(static_cast<int>(threads)) { arena_.initialize(); }

  void run(size_t groups, void (*function)(void*, size_t), void* data) {
    // One task per work-group (grain 1, simple partitioner): the work-items claim partitions
    // dynamically, so what each does does not matter for the result, but each must get a task of its own
    // or an idle thread would find a sibling's whole share already claimed.
    arena_.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, groups, 1),
          [&](const tbb::blocked_range<size_t>& range) {
            for (size_t group = range.begin(); group != range.end(); ++group) {
              function(data, group);
            }
          },
          tbb::simple_partitioner());
    });
  }

 private:
  tbb::task_arena arena_;
};
#endif

}  // namespace detail

namespace {

std::mutex g_default_mutex;
std::shared_ptr<ExecutionContext> g_default_context;

}  // namespace

ExecutionContext::ExecutionContext() : ExecutionContext(0) {}

ExecutionContext::ExecutionContext(size_t threads) : threads_(threads) { createExecutor(); }

ExecutionContext::~ExecutionContext() = default;

bool ExecutionContext::parallelAvailable() {
#if MOPTIM_HAS_TBB
  return true;
#else
  return false;
#endif
}

void ExecutionContext::setThreads(size_t threads) {
  threads_ = threads;
  createExecutor();
}

size_t ExecutionContext::resolvedThreads() const {
#if MOPTIM_HAS_TBB
  return threads_ != 0 ? threads_ : std::max<size_t>(1, static_cast<size_t>(tbb::info::default_concurrency()));
#else
  return 1;
#endif
}

void ExecutionContext::createExecutor() {
#if MOPTIM_HAS_TBB
  executor_ = std::make_unique<detail::HostExecutor>(resolvedThreads());
#endif
}

size_t ExecutionContext::launchWidth(size_t partitions, LaunchWork work) const {
  if (partitions == 0) {
    return 0;
  }
#if MOPTIM_HAS_TBB
  const size_t minimum = work == LaunchWork::CostOnly ? kMinPartitionsToGoWideCostOnly : kMinPartitionsToGoWide;
  if (partitions < minimum) {
    return 1;
  }
  constexpr size_t kPartitionsPerWorkItem = 2;
  return std::max<size_t>(
      1, std::min(resolvedThreads(), (partitions + kPartitionsPerWorkItem - 1) / kPartitionsPerWorkItem));
#else
  (void)work;
  return 1;
#endif
}

void ExecutionContext::runOnHost(size_t groups, void (*function)(void*, size_t), void* data) {
#if MOPTIM_HAS_TBB
  if (groups > 1 && executor_) {
    executor_->run(groups, function, data);
    return;
  }
#endif
  for (size_t group = 0; group < groups; ++group) {
    function(data, group);
  }
}

std::shared_ptr<ExecutionContext> defaultContext() {
  const std::lock_guard lock(g_default_mutex);
  if (!g_default_context) {
    g_default_context = std::make_shared<ExecutionContext>();
  }
  return g_default_context;
}

void setDefaultContext(std::shared_ptr<ExecutionContext> context) {
  const std::lock_guard lock(g_default_mutex);
  g_default_context = std::move(context);
}

}  // namespace moptim
