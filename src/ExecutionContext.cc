#include "moptim/ExecutionContext.hh"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace moptim {
namespace detail {

/**
 * Persistent host threads for the host backend's wide launches. The caller takes part: a launch of
 * `groups` work-groups is shared between the caller and the workers, which claim work-groups through
 * an atomic counter, and run() returns when every work-group is done. Launches are serialized.
 */
class HostPool {
 public:
  explicit HostPool(size_t workers) {
    threads_.reserve(workers);
    for (size_t i = 0; i < workers; ++i) {
      threads_.emplace_back([this] { workerLoop(); });
    }
  }

  ~HostPool() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread& thread : threads_) {
      thread.join();
    }
  }

  HostPool(const HostPool&) = delete;
  HostPool& operator=(const HostPool&) = delete;

  size_t workers() const { return threads_.size(); }

  void run(size_t groups, void (*function)(void*, size_t), void* data) {
    const std::lock_guard serialize(run_mutex_);
    {
      const std::lock_guard lock(mutex_);
      function_ = function;
      data_ = data;
      groups_ = groups;
      next_.store(0, std::memory_order_relaxed);
      busy_ = threads_.size();
      error_ = nullptr;
      ++generation_;
    }
    start_.notify_all();
    work();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return busy_ == 0; });
    if (error_) {
      std::rethrow_exception(std::exchange(error_, nullptr));
    }
  }

 private:
  void work() {
    for (size_t group = next_.fetch_add(1, std::memory_order_relaxed); group < groups_;
         group = next_.fetch_add(1, std::memory_order_relaxed)) {
      try {
        function_(data_, group);
      } catch (...) {
        const std::lock_guard lock(mutex_);
        if (!error_) {
          error_ = std::current_exception();
        }
      }
    }
  }

  void workerLoop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
      }
      work();
      {
        const std::lock_guard lock(mutex_);
        --busy_;
      }
      done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  uint64_t generation_ = 0;
  bool stop_ = false;
  void (*function_)(void*, size_t) = nullptr;
  void* data_ = nullptr;
  size_t groups_ = 0;
  std::atomic<size_t> next_{0};
  size_t busy_ = 0;
  std::exception_ptr error_;
};

}  // namespace detail

namespace {

#if MOPTIM_HAS_SYCL
sycl::queue makeDefaultQueue() {
  // Without a handler the SYCL runtime terminates the process on an asynchronous error; this one
  // hands the first error to the wait_and_throw() of detail::complete().
  const auto rethrow = [](sycl::exception_list errors) {
    for (const std::exception_ptr& error : errors) {
      std::rethrow_exception(error);
    }
  };
  return sycl::queue{sycl::default_selector_v, rethrow, sycl::property_list{sycl::property::queue::in_order()}};
}
#endif

size_t hardwareThreads() { return std::max<size_t>(1, std::thread::hardware_concurrency()); }

/// On the host, a work-group gets at least this many partitions: a cost of one or two partitions (a
/// thousand point-to-plane elements) then runs on the calling thread.
constexpr size_t kHostPartitionsPerWorkGroup = 2;

std::mutex g_default_mutex;
std::shared_ptr<ExecutionContext> g_default_context;

}  // namespace

Backend defaultBackend() {
  const char* requested = std::getenv("MOPTIM_BACKEND");
  if (requested != nullptr && std::string_view(requested) == "host") {
    return Backend::Host;
  }
  return syclAvailable() ? Backend::Sycl : Backend::Host;
}

const char* backendName(Backend backend) { return backend == Backend::Sycl ? "sycl" : "host"; }

ExecutionContext::ExecutionContext() : ExecutionContext(defaultBackend()) {}

ExecutionContext::ExecutionContext(Backend backend, size_t threads, AssemblyStrategy strategy)
    : backend_(Backend::Host), threads_(threads), strategy_(strategy), compute_units_(hardwareThreads()) {
  if (backend == Backend::Sycl) {
#if MOPTIM_HAS_SYCL
    initSycl(makeDefaultQueue());
#else
    throw std::invalid_argument("moptim: the Sycl backend needs a build with MOPTIM_WITH_SYCL");
#endif
  }
}

#if MOPTIM_HAS_SYCL
ExecutionContext::ExecutionContext(sycl::queue queue, size_t threads, AssemblyStrategy strategy)
    : backend_(Backend::Sycl), threads_(threads), strategy_(strategy) {
  initSycl(std::move(queue));
}

void ExecutionContext::initSycl(sycl::queue queue) {
  if (!queue.is_in_order()) {
    throw std::invalid_argument("ExecutionContext: the queue must be in-order");
  }
  backend_ = Backend::Sycl;
  queue_ = std::move(queue);
  const sycl::device device = queue_->get_device();
  compute_units_ = std::max<size_t>(1, device.get_info<sycl::info::device::max_compute_units>());
  max_work_group_size_ = std::max<size_t>(1, device.get_info<sycl::info::device::max_work_group_size>());
  local_memory_bytes_ = device.get_info<sycl::info::device::local_mem_size>();
  host_accessible_ = device.has(sycl::aspect::usm_system_allocations);
  is_cpu_ = device.is_cpu();
}
#endif

ExecutionContext::~ExecutionContext() = default;

size_t ExecutionContext::launchWidth(size_t partitions) const {
  if (partitions == 0) {
    return 0;
  }
  if (backend_ == Backend::Host) {
    const size_t useful = (partitions + kHostPartitionsPerWorkGroup - 1) / kHostPartitionsPerWorkGroup;
    return std::max<size_t>(1, std::min(resolvedThreads(), useful));
  }
  return std::min(resolvedThreads(), partitions);
}

void* ExecutionContext::allocateShared(size_t bytes, size_t alignment) {
  bytes = std::max<size_t>(bytes, 1);
#if MOPTIM_HAS_SYCL
  if (backend_ == Backend::Sycl) {
    void* ptr = sycl::aligned_alloc_shared(alignment, bytes, *queue_);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return ptr;
  }
#endif
  // std::aligned_alloc wants a size that is a multiple of the alignment.
  void* ptr = std::aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void* ExecutionContext::allocateDevice(size_t bytes, size_t alignment) {
#if MOPTIM_HAS_SYCL
  if (backend_ == Backend::Sycl) {
    void* ptr = sycl::aligned_alloc_device(alignment, std::max<size_t>(bytes, 1), *queue_);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return ptr;
  }
#endif
  return allocateShared(bytes, alignment);
}

void ExecutionContext::deallocate(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
#if MOPTIM_HAS_SYCL
  if (backend_ == Backend::Sycl) {
    sycl::free(ptr, *queue_);
    return;
  }
#endif
  std::free(ptr);
}

void ExecutionContext::copyToDevice(void* destination, const void* source, size_t bytes) {
#if MOPTIM_HAS_SYCL
  if (backend_ == Backend::Sycl) {
    queue_->memcpy(destination, source, bytes).wait();
    return;
  }
#endif
  std::memcpy(destination, source, bytes);
}

void ExecutionContext::runOnHost(size_t groups, void (*function)(void*, size_t), void* data) {
  const size_t threads = std::min(resolvedThreads(), groups);
  if (threads <= 1) {
    for (size_t group = 0; group < groups; ++group) {
      function(data, group);
    }
    return;
  }
  if (!pool_ || pool_->workers() + 1 < threads) {
    pool_ = std::make_unique<detail::HostPool>(resolvedThreads() - 1);
  }
  pool_->run(groups, function, data);
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
