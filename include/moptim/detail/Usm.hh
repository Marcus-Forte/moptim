#pragma once

#include <cstddef>
#include <memory>
#include <new>

#include "moptim/ExecutionContext.hh"

namespace moptim::detail {

/// Frees memory of an ExecutionContext (USM on a SYCL context, host memory on a host one). The context
/// must outlive the allocation (the costs declare their context before their buffers).
struct UsmDeleter {
  ExecutionContext* context = nullptr;
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr && context != nullptr) {
      context->deallocate(ptr);
    }
  }
};

template <class U>
using UsmPtr = std::unique_ptr<U, UsmDeleter>;

/// Default alignment of the helpers below: a cache line, which also covers every SIMD width.
inline constexpr size_t kUsmAlignment = 64;

/// Allocates `count` elements (at least one) that the host and the kernels can access, uninitialized,
/// aligned to `alignment`. Throws std::bad_alloc on failure.
template <class U>
UsmPtr<U> makeSharedArray(ExecutionContext& context, size_t count, size_t alignment = kUsmAlignment) {
  void* ptr = context.allocateShared((count > 0 ? count : 1) * sizeof(U), alignment);
  return UsmPtr<U>(static_cast<U*>(ptr), UsmDeleter{&context});
}

/// Allocates `count` elements (at least one) that the kernels can access, uninitialized, aligned to
/// `alignment`; not necessarily accessible from the host. Throws std::bad_alloc on failure.
template <class U>
UsmPtr<U> makeDeviceArray(ExecutionContext& context, size_t count, size_t alignment = kUsmAlignment) {
  void* ptr = context.allocateDevice((count > 0 ? count : 1) * sizeof(U), alignment);
  return UsmPtr<U>(static_cast<U*>(ptr), UsmDeleter{&context});
}

}  // namespace moptim::detail
