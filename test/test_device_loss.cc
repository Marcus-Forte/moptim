// The built-in losses evaluated by a kernel from their descriptors against their host evaluate().
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "moptim/ExecutionContext.hh"
#include "moptim/LossFunction.hh"
#include "moptim/detail/Usm.hh"

using namespace moptim;

namespace {

/// Distance in representable values between two finite numbers of the same sign.
template <class T>
uint64_t ulpDistance(T a, T b) {
  using Bits = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
  Bits x, y;
  std::memcpy(&x, &a, sizeof(T));
  std::memcpy(&y, &b, sizeof(T));
  // Map the sign-magnitude representation to an ordered one.
  const Bits sign = Bits{1} << (sizeof(T) * 8 - 1);
  const auto ordered = [&](Bits bits) { return (bits & sign) ? ~bits + 1 : bits | sign; };
  const Bits ox = ordered(x), oy = ordered(y);
  return static_cast<uint64_t>(ox > oy ? ox - oy : oy - ox);
}

template <class T>
std::vector<T> sampleValues(T scale) {
  std::vector<T> values{T{0}, std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::min(),
                        std::numeric_limits<T>::epsilon() * scale * scale, T{1e-12}, T{1e-6}};
  for (const T factor : {T{0.5}, T{0.9}, T{0.999}, T{1}, T{1.001}, T{1.1}, T{2}, T{10}, T{100}, T{1e4}}) {
    values.push_back(factor * scale * scale);  // s = ||r||^2 around, below and above the scale
    values.push_back(factor);
  }
  values.push_back(std::numeric_limits<T>::max() / 1e4);
  // A dense sweep.
  for (int i = 1; i <= 400; ++i) {
    values.push_back(static_cast<T>(i) * scale * scale / 50);
  }
  return values;
}

template <class T>
void checkLoss(const LossFunction<T>& loss, uint64_t max_ulp_of_rho, const char* name) {
  ExecutionContext context(sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()});
  const LossDescriptor<T> descriptor = loss.descriptor();
  ASSERT_NE(descriptor.kind, LossKind::Custom) << name;

  const std::vector<T> values = sampleValues<T>(T{1.5});
  const size_t n = values.size();
  auto s = detail::makeSharedArray<T>(context, n);
  auto out = detail::makeSharedArray<T>(context, 3 * n);
  std::copy(values.begin(), values.end(), s.get());
  const T* s_ptr = s.get();
  T* out_ptr = out.get();
  detail::launch(context, 1, [=](size_t) {
    for (size_t i = 0; i < n; ++i) {
      detail::evaluateLoss(descriptor, s_ptr[i], out_ptr + 3 * i);
    }
  });

  for (size_t i = 0; i < n; ++i) {
    T host[3];
    loss.evaluate(values[i], host);
    for (int k = 0; k < 3; ++k) {
      const T device = out_ptr[3 * i + k];
      if (std::isnan(host[k]) || std::isinf(host[k])) {
        EXPECT_EQ(std::isnan(host[k]), std::isnan(device)) << name << " s=" << values[i] << " out[" << k << "]";
        if (std::isinf(host[k])) EXPECT_EQ(host[k], device) << name << " s=" << values[i] << " out[" << k << "]";
        continue;
      }
      const uint64_t allowed = k == 0 ? max_ulp_of_rho : 0;
      EXPECT_LE(ulpDistance(host[k], device), allowed)
          << name << " s=" << values[i] << " out[" << k << "] host " << host[k] << " device " << device;
    }
  }
}

}  // namespace

template <class T>
class DeviceLossTest : public ::testing::Test {};
using Scalars = ::testing::Types<double, float>;
TYPED_TEST_SUITE(DeviceLossTest, Scalars);

TYPED_TEST(DeviceLossTest, Trivial) { checkLoss<TypeParam>(TrivialLoss<TypeParam>{}, 0, "trivial"); }
TYPED_TEST(DeviceLossTest, Huber) { checkLoss<TypeParam>(HuberLoss<TypeParam>{TypeParam{1.5}}, 0, "huber"); }
// rho of the Cauchy loss goes through std::log, which the host and the device may round differently.
TYPED_TEST(DeviceLossTest, Cauchy) { checkLoss<TypeParam>(CauchyLoss<TypeParam>{TypeParam{1.5}}, 4, "cauchy"); }
TYPED_TEST(DeviceLossTest, GemanMcClure) {
  checkLoss<TypeParam>(GemanMcClureLoss<TypeParam>{TypeParam{1.5}}, 0, "geman-mcclure");
}

TEST(TestLossDescriptor, ScaleChangesAreReflected) {
  HuberLoss<double> huber(1.0);
  huber.setScale(3.0);
  EXPECT_EQ(huber.descriptor().kind, LossKind::Huber);
  EXPECT_EQ(huber.descriptor().p0, 3.0);
  EXPECT_EQ(huber.descriptor().p1, 9.0);

  CauchyLoss<double> cauchy(1.0);
  cauchy.setScale(2.0);
  EXPECT_EQ(cauchy.descriptor().kind, LossKind::Cauchy);
  EXPECT_EQ(cauchy.descriptor().p0, 4.0);

  GemanMcClureLoss<float> geman(1.0f);
  geman.setScale(0.25f);
  EXPECT_EQ(geman.descriptor().kind, LossKind::GemanMcClure);
  EXPECT_EQ(geman.descriptor().p0, 0.25f);

  EXPECT_EQ(TrivialLoss<double>{}.descriptor().kind, LossKind::Trivial);
}

TEST(TestLossDescriptor, CustomLossIsCustom) {
  struct Squared : LossFunction<double> {
    void evaluate(double s, double out[3]) const override { out[0] = out[1] = out[2] = s; }
  };
  EXPECT_EQ(Squared{}.descriptor().kind, LossKind::Custom);
  double out[3];
  detail::evaluateLoss(Squared{}.descriptor(), 1.0, out);
  EXPECT_TRUE(std::isnan(out[0]));
}
