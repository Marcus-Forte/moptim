#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace moptim::detail {

/// Frees what makeAlignedArray() allocated.
struct AlignedDeleter {
  void operator()(void* ptr) const noexcept { std::free(ptr); }
};

template <class U>
using AlignedPtr = std::unique_ptr<U, AlignedDeleter>;

/// Bytes in a cache line; also the unit the costs pad shared counters and partial slots to.
inline constexpr size_t kDefaultAlignment = 64;

/// Allocates `count` elements (at least one) of plain heap memory, uninitialized, aligned to
/// `alignment` (a power of two; a cache line by default, which also covers every SIMD width). The
/// elements are not constructed: the caller places objects into it. Throws std::bad_alloc on failure.
template <class U>
AlignedPtr<U> makeAlignedArray(size_t count, size_t alignment = kDefaultAlignment) {
  const size_t bytes = (count > 0 ? count : 1) * sizeof(U);
  // std::aligned_alloc wants a size that is a multiple of the alignment.
  void* ptr = std::aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return AlignedPtr<U>(static_cast<U*>(ptr));
}

}  // namespace moptim::detail
