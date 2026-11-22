#pragma once

// Relaxed atomic operations on plain memory for code that runs in kernels on both backends: in the device
// pass of a SYCL build they use sycl::atomic_ref (device scope), everywhere else std::atomic_ref, so the
// same function body works as a SYCL kernel and as a host work-group.
#include <atomic>
#include <cstddef>
#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
#endif

namespace moptim {

#ifdef __SYCL_DEVICE_ONLY__
template <class T>
using AtomicRef = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>;
#endif

/// Atomically adds `value` to `object` and returns the previous value.
template <class T>
T atomicFetchAdd(T& object, T value) {
#ifdef __SYCL_DEVICE_ONLY__
  return AtomicRef<T>(object).fetch_add(value);
#else
  return std::atomic_ref<T>(object).fetch_add(value, std::memory_order_relaxed);
#endif
}

/// Atomically replaces `object` by `min(object, value)` and returns the previous value.
template <class T>
T atomicFetchMin(T& object, T value) {
#ifdef __SYCL_DEVICE_ONLY__
  return AtomicRef<T>(object).fetch_min(value);
#else
  std::atomic_ref<T> ref(object);
  T current = ref.load(std::memory_order_relaxed);
  while (value < current && !ref.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
  return current;
#endif
}

/// If `object` equals `expected`, replaces it by `desired` and returns true; otherwise loads it into
/// `expected` and returns false.
template <class T>
bool atomicCompareExchange(T& object, T& expected, T desired) {
#ifdef __SYCL_DEVICE_ONLY__
  return AtomicRef<T>(object).compare_exchange_strong(expected, desired);
#else
  return std::atomic_ref<T>(object).compare_exchange_strong(expected, desired, std::memory_order_relaxed);
#endif
}

template <class T>
T atomicLoad(T& object) {
#ifdef __SYCL_DEVICE_ONLY__
  return AtomicRef<T>(object).load();
#else
  return std::atomic_ref<T>(object).load(std::memory_order_relaxed);
#endif
}

template <class T>
void atomicStore(T& object, T value) {
#ifdef __SYCL_DEVICE_ONLY__
  AtomicRef<T>(object).store(value);
#else
  std::atomic_ref<T>(object).store(value, std::memory_order_relaxed);
#endif
}

}  // namespace moptim
