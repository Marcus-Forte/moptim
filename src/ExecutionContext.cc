#include "moptim/ExecutionContext.hh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace moptim {
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
    : backend_(Backend::Host), threads_(threads), strategy_(strategy) {
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
    return 1;
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
