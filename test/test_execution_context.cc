#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "moptim/ExecutionContext.hh"
#include "moptim/detail/Usm.hh"

using namespace moptim;

namespace {

/// Puts the process-wide context back after a test that replaced it.
class RestoreDefaultContext {
 public:
  RestoreDefaultContext() : saved_(defaultContext()) {}
  ~RestoreDefaultContext() { setDefaultContext(saved_); }

 private:
  std::shared_ptr<ExecutionContext> saved_;
};

sycl::queue inOrderQueue() { return sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()}; }

}  // namespace

TEST(TestExecutionContext, DefaultContextExistsAndIsInOrder) {
  const std::shared_ptr<ExecutionContext> context = defaultContext();
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(context->queue().is_in_order());
  EXPECT_EQ(context, defaultContext());
  EXPECT_EQ(context->threads(), 0u);
  EXPECT_EQ(context->strategy(), AssemblyStrategy::Auto);
}

TEST(TestExecutionContext, SetDefaultContext) {
  const RestoreDefaultContext restore;
  const auto custom = std::make_shared<ExecutionContext>(inOrderQueue(), 2, AssemblyStrategy::Chunked);
  setDefaultContext(custom);
  EXPECT_EQ(defaultContext(), custom);
  EXPECT_EQ(defaultContext()->threads(), 2u);
  EXPECT_EQ(defaultContext()->strategy(), AssemblyStrategy::Chunked);

  // nullptr discards it; the next call creates a fresh one.
  setDefaultContext(nullptr);
  const std::shared_ptr<ExecutionContext> fresh = defaultContext();
  ASSERT_NE(fresh, nullptr);
  EXPECT_NE(fresh, custom);
}

TEST(TestExecutionContext, RejectsOutOfOrderQueue) {
  EXPECT_THROW(ExecutionContext(sycl::queue{sycl::default_selector_v}), std::invalid_argument);
}

TEST(TestExecutionContext, Knobs) {
  ExecutionContext context(inOrderQueue());
  context.setThreads(3);
  EXPECT_EQ(context.threads(), 3u);
  context.setStrategy(AssemblyStrategy::WorkGroup);
  EXPECT_EQ(context.strategy(), AssemblyStrategy::WorkGroup);
  // Only reports the device; the values depend on it.
  RecordProperty("host_accessible", context.hostAccessible());
  RecordProperty("is_cpu", context.isCpu());
}

TEST(TestExecutionContext, LaunchWidth) {
  ExecutionContext context(inOrderQueue());
  const size_t units = context.queue().get_device().get_info<sycl::info::device::max_compute_units>();

  // threads == 0: all compute units, never more than the partitions.
  EXPECT_EQ(context.launchWidth(0), 0u);
  EXPECT_EQ(context.launchWidth(1), 1u);
  EXPECT_EQ(context.launchWidth(units), units);
  EXPECT_EQ(context.launchWidth(units + 5), units);
  EXPECT_EQ(context.launchWidth(100000), units);

  context.setThreads(3);
  EXPECT_EQ(context.launchWidth(0), 0u);
  EXPECT_EQ(context.launchWidth(1), 1u);
  EXPECT_EQ(context.launchWidth(2), 2u);
  EXPECT_EQ(context.launchWidth(3), 3u);
  EXPECT_EQ(context.launchWidth(100000), 3u);

  // More threads than compute units are honoured, but never exceed the partitions.
  context.setThreads(1000);
  EXPECT_EQ(context.launchWidth(5), 5u);
  EXPECT_EQ(context.launchWidth(5000), 1000u);

  for (const size_t threads : {size_t{0}, size_t{1}, size_t{4}, size_t{64}}) {
    context.setThreads(threads);
    for (size_t partitions = 1; partitions < 200; ++partitions) {
      const size_t width = context.launchWidth(partitions);
      EXPECT_GE(width, 1u);
      EXPECT_LE(width, partitions);
    }
  }
}

TEST(TestExecutionContext, LaunchCounterCountsLaunches) {
  ExecutionContext context(inOrderQueue());
  auto flag = detail::makeSharedArray<int>(context.queue(), 1);
  int* out = flag.get();
  *out = 0;
  EXPECT_EQ(context.launches(), 0u);

  detail::launch(context, 1, [=](sycl::nd_item<1>) { *out = 42; });
  EXPECT_EQ(context.launches(), 1u);
  EXPECT_EQ(*out, 42);

  detail::launch(context, 1, [=](sycl::nd_item<1>) { *out = 43; });
  EXPECT_EQ(context.launches(), 2u);
  EXPECT_EQ(*out, 43);

  // Nothing to run: no launch.
  detail::launch(context, 0, [=](sycl::nd_item<1>) { *out = -1; });
  EXPECT_EQ(context.launches(), 2u);
  EXPECT_EQ(*out, 43);
}

TEST(TestExecutionContext, LaunchRunsEveryWorkItem) {
  ExecutionContext context(inOrderQueue());
  for (const size_t groups : {size_t{1}, size_t{2}, size_t{7}, size_t{64}}) {
    auto slots = detail::makeSharedArray<size_t>(context.queue(), groups);
    size_t* out = slots.get();
    std::fill(out, out + groups, size_t{0});
    detail::launch(context, groups, [=](sycl::nd_item<1> item) { out[item.get_group_linear_id()] += 100 + item.get_group_linear_id(); });
    for (size_t g = 0; g < groups; ++g) {
      EXPECT_EQ(out[g], 100 + g) << "groups " << groups << ", group " << g;
    }
  }
}

TEST(TestExecutionContext, UsmHelpersAlign) {
  sycl::queue queue = inOrderQueue();
  auto shared = detail::makeSharedArray<double>(queue, 5, 128);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(shared.get()) % 128, 0u);
  auto empty = detail::makeSharedArray<double>(queue, 0);
  EXPECT_NE(empty.get(), nullptr);
  auto device = detail::makeDeviceArray<float>(queue, 16);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(device.get()) % detail::kUsmAlignment, 0u);
}

// Fixed-size Eigen members are shared between the host and the device pass of a -fsycl compilation
// (a model is copied into USM), so their layout must agree. The device pass runs scalar Eigen
// (EIGEN_DONT_VECTORIZE, given by the build to the device front end only), the host pass SIMD.
namespace {

struct EigenLayout {
  int vectorized;
  int max_static_align;
  size_t vector4_size, vector4_align;
  size_t affine3_size, affine3_align;
  size_t matrix4_size, matrix4_align;
};

EigenLayout eigenLayoutOfThisPass() {
  EigenLayout layout{};
#ifdef EIGEN_VECTORIZE
  layout.vectorized = 1;
#else
  layout.vectorized = 0;
#endif
  layout.max_static_align = EIGEN_MAX_STATIC_ALIGN_BYTES;
  layout.vector4_size = sizeof(Eigen::Vector4d);
  layout.vector4_align = alignof(Eigen::Vector4d);
  layout.affine3_size = sizeof(Eigen::Affine3d);
  layout.affine3_align = alignof(Eigen::Affine3d);
  layout.matrix4_size = sizeof(Eigen::Matrix4f);
  layout.matrix4_align = alignof(Eigen::Matrix4f);
  return layout;
}

}  // namespace

TEST(TestExecutionContext, EigenSettingsOfHostAndDevicePass) {
  ExecutionContext context(inOrderQueue());
  auto slot = detail::makeSharedArray<EigenLayout>(context.queue(), 1);
  EigenLayout* device_layout = slot.get();
  detail::launch(context, 1, [=](sycl::nd_item<1>) { *device_layout = eigenLayoutOfThisPass(); });
  const EigenLayout host = eigenLayoutOfThisPass();
  const EigenLayout device = *device_layout;

  EXPECT_EQ(host.vectorized, 1) << "the host pass must use Eigen's SIMD code";
  EXPECT_EQ(device.vectorized, 0) << "the device pass must run scalar Eigen (EIGEN_DONT_VECTORIZE)";
  EXPECT_EQ(host.max_static_align, 16);
  EXPECT_EQ(device.max_static_align, 16);
  EXPECT_EQ(host.vector4_size, device.vector4_size);
  EXPECT_EQ(host.vector4_align, device.vector4_align);
  EXPECT_EQ(host.affine3_size, device.affine3_size);
  EXPECT_EQ(host.affine3_align, device.affine3_align);
  EXPECT_EQ(host.matrix4_size, device.matrix4_size);
  EXPECT_EQ(host.matrix4_align, device.matrix4_align);
}
