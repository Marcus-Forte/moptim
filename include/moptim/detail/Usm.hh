#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <sycl/sycl.hpp>

namespace moptim::detail {

/// Frees USM memory on the queue it was allocated with.
struct UsmDeleter {
  std::optional<sycl::queue> queue;  // optional: a default-constructed deleter must not create a queue
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr && queue) {
      sycl::free(ptr, *queue);
    }
  }
};

template <class U>
using UsmPtr = std::unique_ptr<U, UsmDeleter>;

/// Default alignment of the helpers below: a cache line, which also covers every SIMD width.
inline constexpr size_t kUsmAlignment = 64;

/// Allocates `count` elements (at least one) of shared USM, uninitialized, aligned to `alignment`.
/// Throws std::bad_alloc on failure.
template <class U>
UsmPtr<U> makeSharedArray(sycl::queue& queue, size_t count, size_t alignment = kUsmAlignment) {
  void* ptr = sycl::aligned_alloc_shared(alignment, (count > 0 ? count : 1) * sizeof(U), queue);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return UsmPtr<U>(static_cast<U*>(ptr), UsmDeleter{queue});
}

/// Allocates `count` elements (at least one) of device USM, uninitialized, aligned to `alignment`.
/// Not necessarily accessible from the host. Throws std::bad_alloc on failure.
template <class U>
UsmPtr<U> makeDeviceArray(sycl::queue& queue, size_t count, size_t alignment = kUsmAlignment) {
  void* ptr = sycl::aligned_alloc_device(alignment, (count > 0 ? count : 1) * sizeof(U), queue);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return UsmPtr<U>(static_cast<U*>(ptr), UsmDeleter{queue});
}

}  // namespace moptim::detail
