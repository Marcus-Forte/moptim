// The host backend of ExecutionContext: launches on the calling thread or the host pool.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "moptim/Atomic.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/detail/Usm.hh"

using namespace moptim;

namespace {
struct HostKernelName;
}

TEST(HostBackend, EveryWorkGroupRunsOnceAtAnyThreadCount) {
  for (const size_t threads : {size_t{1}, size_t{2}, size_t{3}, size_t{0}}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    ExecutionContext context(Backend::Host, threads);
    EXPECT_EQ(context.backend(), Backend::Host);
    EXPECT_TRUE(context.hostAccessible());
    EXPECT_FALSE(context.usesWorkGroupKernel());
    for (const size_t groups : {size_t{1}, size_t{5}, size_t{37}}) {
      // Device-compatible kernel (in a SYCL build it is compiled as one too): plain memory, atomics.
      auto runs = detail::makeSharedArray<int>(context, groups);
      for (size_t g = 0; g < groups; ++g) {
        runs.get()[g] = 0;
      }
      int* counts = runs.get();
      const uint64_t before = context.launches();
      detail::launchNamed<HostKernelName>(context, groups,
                                          [counts](size_t group) { atomicFetchAdd(counts[group], 1); });
      EXPECT_EQ(context.launches() - before, 1u);
      for (size_t g = 0; g < groups; ++g) {
        ASSERT_EQ(counts[g], 1) << g;
      }
    }
  }
}

TEST(HostBackend, SmallLaunchesStayOnTheCallingThread) {
  ExecutionContext context(Backend::Host, 8);
  EXPECT_EQ(context.launchWidth(0), 0u);
  EXPECT_EQ(context.launchWidth(1), 1u);
  EXPECT_EQ(context.launchWidth(2), 1u);  // a cost of about a thousand point-to-plane elements
  EXPECT_EQ(context.launchWidth(3), 2u);
  EXPECT_EQ(context.launchWidth(1000), 8u);
  context.setThreads(1);
  EXPECT_EQ(context.launchWidth(1000), 1u);
}

TEST(HostBackend, AnExceptionOfAWorkGroupReachesTheCaller) {
  ExecutionContext context(Backend::Host, 4);
  auto throwing = [](size_t group) {
    if (group == 11) {
      throw std::runtime_error("work-group 11");
    }
  };
  EXPECT_THROW(detail::runKernelOnHost(context, 16, throwing), std::runtime_error);
  // The pool is still usable.
  std::atomic<int> total{0};
  auto counting = [&total](size_t) { total.fetch_add(1); };
  detail::runKernelOnHost(context, 16, counting);
  EXPECT_EQ(total.load(), 16);
}

TEST(HostBackend, MemoryIsAligned) {
  ExecutionContext context(Backend::Host);
  auto shared = detail::makeSharedArray<double>(context, 5, 128);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(shared.get()) % 128, 0u);
  auto empty = detail::makeSharedArray<double>(context, 0);
  EXPECT_NE(empty.get(), nullptr);
  auto device = detail::makeDeviceArray<float>(context, 16);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(device.get()) % detail::kUsmAlignment, 0u);
  const double source[3] = {1.0, 2.0, 3.0};
  context.copyToDevice(shared.get(), source, sizeof(source));
  EXPECT_EQ(shared.get()[2], 3.0);
}

TEST(HostBackend, SyclNeedsASyclBuild) {
  if (syclAvailable()) {
    GTEST_SKIP() << "this build has SYCL";
  }
  EXPECT_THROW((void)ExecutionContext{Backend::Sycl}, std::invalid_argument);
  EXPECT_EQ(defaultBackend(), Backend::Host);
}
