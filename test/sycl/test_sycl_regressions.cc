#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <span>
#include <vector>

#include "moptim/NumericalCostForwardEuler.hh"
#include "moptim/NumericalCostSycl.hh"

using namespace moptim;

namespace {

/// 2 inputs, 1 observation, 4 parameters: r = p0*x0 + p1*x1 + p2*x0*x1 + p3 - y
template <class T>
struct PlaneModel {
  void setState(const T*) {}
  void residual(const T* x, const T* in, const T* obs, T* res) {
    res[0] = x[0] * in[0] + x[1] * in[1] + x[2] * in[0] * in[1] + x[3] - obs[0];
  }
};

/// Same as PlaneModel but with a configurable offset set via the constructor.
template <class T>
struct OffsetModel {
  T offset{0};
  void setState(const T*) {}
  void residual(const T* x, const T* in, const T* obs, T* res) { res[0] = x[0] * in[0] + offset - obs[0]; }
};

template <class T>
struct Data {
  std::vector<T> in, obs;
  Data(size_t n, size_t in_dim, size_t obs_dim) : in(n * in_dim), obs(n * obs_dim) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : in) v = static_cast<T>(dist(gen));
    for (auto& v : obs) v = static_cast<T>(dist(gen));
  }
};

template <class T>
void expectClose(const T* a, const T* b, size_t n, double rel) {
  double scale = 1e-30;
  for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(static_cast<double>(b[i])));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(static_cast<double>(a[i]), static_cast<double>(b[i]), rel * scale) << "index " << i;
  }
}

template <class T, class Model>
void compareWithCpu(size_t n, size_t in_dim, size_t obs_dim, size_t param_dim, size_t active, const T* x,
                    Model model = Model{}) {
  Data<T> data(n, in_dim, obs_dim);
  sycl::queue queue{sycl::default_selector_v};
  NumericalCostSycl<T, Model> sycl_cost(queue, std::span<const T>(data.in), std::span<const T>(data.obs), in_dim,
                                        obs_dim, param_dim, n, active, model);
  NumericalCostForwardEuler<Model, T> cpu_cost(data.in.data(), data.obs.data(), n, in_dim, obs_dim, param_dim, model,
                                               active);

  std::vector<T> jtj_s(param_dim * param_dim, T{-1}), jtj_c(param_dim * param_dim, T{-1});
  std::vector<T> jtb_s(param_dim, T{-1}), jtb_c(param_dim, T{-1});
  T cost_s = 0, cost_c = 0;
  sycl_cost.computeLinearSystem(x, jtj_s.data(), jtb_s.data(), cost_s);
  cpu_cost.computeLinearSystem(x, jtj_c.data(), jtb_c.data(), cost_c);

  const double rel = std::is_same_v<T, double> ? 1e-9 : 1e-3;
  EXPECT_NEAR(cost_s, cost_c, rel * std::abs(cost_c));
  EXPECT_NEAR(sycl_cost.computeCost(x), cpu_cost.computeCost(x), rel * std::abs(cost_c));
  expectClose(jtj_s.data(), jtj_c.data(), jtj_c.size(), rel);
  expectClose(jtb_s.data(), jtb_c.data(), jtb_c.size(), rel);
}

}  // namespace

TEST(TestSyclRegressions, InputDimDifferentFromObservationDim) {
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  compareWithCpu<double, PlaneModel<double>>(1000, 2, 1, 4, 0, x);
}

TEST(TestSyclRegressions, InputDimDifferentFromObservationDimFloat) {
  const float x[4]{0.3f, -0.2f, 0.5f, 0.1f};
  compareWithCpu<float, PlaneModel<float>>(1000, 2, 1, 4, 0, x);
}

TEST(TestSyclRegressions, ActiveParamDimLeavesInactiveBlockZero) {
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  compareWithCpu<double, PlaneModel<double>>(1000, 2, 1, 4, 2, x);
}

TEST(TestSyclRegressions, ConfiguredModelIsUsed) {
  const double x[4]{0.3, 0.0, 0.0, 0.0};
  OffsetModel<double> model;
  model.offset = 0.75;
  compareWithCpu<double, OffsetModel<double>>(500, 2, 1, 1, 0, x, model);
}

TEST(TestSyclRegressions, RepeatedCallsAreIdentical) {
  Data<double> data(2000, 2, 1);
  sycl::queue queue{sycl::default_selector_v};
  NumericalCostSycl<double, PlaneModel<double>> cost(queue, std::span<const double>(data.in),
                                                     std::span<const double>(data.obs), 2, 1, 4, 2000);
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  std::vector<double> jtj1(16), jtj2(16), jtb1(4), jtb2(4);
  double c1 = 0, c2 = 0;
  cost.computeLinearSystem(x, jtj1.data(), jtb1.data(), c1);
  cost.computeLinearSystem(x, jtj2.data(), jtb2.data(), c2);
  EXPECT_NEAR(c1, c2, 1e-9 * std::abs(c1));
  expectClose(jtj1.data(), jtj2.data(), 16, 1e-12);
  expectClose(jtb1.data(), jtb2.data(), 4, 1e-12);
}

TEST(TestSyclRegressions, TooSmallSpansThrow) {
  Data<double> data(10, 2, 1);
  sycl::queue queue{sycl::default_selector_v};
  EXPECT_THROW((NumericalCostSycl<double, PlaneModel<double>>(queue, std::span<const double>(data.in).first(5),
                                                              std::span<const double>(data.obs), 2, 1, 4, 10)),
               std::invalid_argument);
}
