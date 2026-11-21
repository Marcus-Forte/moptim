#include "moptim/ExecutionContext.hh"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace moptim {
namespace {

sycl::queue makeDefaultQueue() {
  // Without a handler the SYCL runtime terminates the process on an asynchronous error; this one
  // hands the first error to the wait_and_throw() of detail::launch().
  const auto rethrow = [](sycl::exception_list errors) {
    for (const std::exception_ptr& error : errors) {
      std::rethrow_exception(error);
    }
  };
  return sycl::queue{sycl::default_selector_v, rethrow, sycl::property_list{sycl::property::queue::in_order()}};
}

std::mutex g_default_mutex;
std::shared_ptr<ExecutionContext> g_default_context;

}  // namespace

ExecutionContext::ExecutionContext() : ExecutionContext(makeDefaultQueue()) {}

ExecutionContext::ExecutionContext(sycl::queue queue, size_t threads, AssemblyStrategy strategy)
    : queue_(std::move(queue)), threads_(threads), strategy_(strategy) {
  if (!queue_.is_in_order()) {
    throw std::invalid_argument("ExecutionContext: the queue must be in-order");
  }
  const sycl::device device = queue_.get_device();
  compute_units_ = std::max<size_t>(1, device.get_info<sycl::info::device::max_compute_units>());
  host_accessible_ = device.has(sycl::aspect::usm_system_allocations);
  is_cpu_ = device.is_cpu();
}

size_t ExecutionContext::launchWidth(size_t partitions) const {
  const size_t width = threads_ != 0 ? threads_ : compute_units_;
  return std::min(width, partitions);
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
