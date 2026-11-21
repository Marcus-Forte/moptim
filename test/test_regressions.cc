// Regression checks that were written for the removed SYCL cost (NumericalCostSycl), ported to the
// numerical forward-difference cost. Where there is a jacobian, the numerical result is compared with
// the analytical one, which is exact for these models.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "transform2d.hh"
#include "transform3d.hh"

using namespace moptim;

namespace {

/// 2 inputs, 1 observation, 4 parameters: r = p0*x0 + p1*x1 + p2*x0*x1 + p3 - y
template <class T>
struct PlaneModel {
  size_t active{4};  // leading parameters the cost differentiates: jacobian() writes no more than that
  void setState(const T*) {}
  void residual(const T* x, const T* in, const T* obs, T* res) {
    res[0] = x[0] * in[0] + x[1] * in[1] + x[2] * in[0] * in[1] + x[3] - obs[0];
  }
  void jacobian(const T*, const T* in, const T*, T* jac) {
    const T full[4]{in[0], in[1], in[0] * in[1], T{1}};
    std::copy(full, full + active, jac);
  }
};

/// Same idea with state set through the constructor: r = p0*x0 + offset - y.
template <class T>
struct OffsetModel {
  T offset{0};
  void setState(const T*) {}
  void residual(const T* x, const T* in, const T* obs, T* res) { res[0] = x[0] * in[0] + offset - obs[0]; }
  void jacobian(const T*, const T* in, const T*, T* jac) { jac[0] = in[0]; }
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
struct System {
  std::vector<T> jtj, jtb;
  T cost{};
  System(size_t params) : jtj(params * params, T{-1}), jtb(params, T{-1}) {}
};

template <class T>
void expectClose(const std::vector<T>& a, const std::vector<T>& b, double rel, const char* what) {
  ASSERT_EQ(a.size(), b.size());
  double scale = 1e-30;
  for (const T v : b) scale = std::max(scale, std::abs(static_cast<double>(v)));
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(a[i]), static_cast<double>(b[i]), rel * scale) << what << " index " << i;
  }
}

template <class T>
double tolerance() {
  return std::is_same_v<T, double> ? 1e-6 : 3e-2;  // forward difference with h = sqrt(eps)
}

template <class T, class Model>
void compareNumericalWithAnalytical(size_t n, size_t in_dim, size_t obs_dim, size_t param_dim, size_t active,
                                    const T* x, Model model = Model{}) {
  const Data<T> data(n, in_dim, obs_dim);
  NumericalCostForwardEuler<Model, T> numerical(data.in.data(), data.obs.data(), n, in_dim, obs_dim, param_dim, model,
                                                active);
  AnalyticalCost<Model, T> analytical(data.in.data(), data.obs.data(), n, in_dim, obs_dim, param_dim, model, active);

  System<T> num(param_dim), ana(param_dim);
  numerical.computeLinearSystem(x, num.jtj.data(), num.jtb.data(), num.cost);
  analytical.computeLinearSystem(x, ana.jtj.data(), ana.jtb.data(), ana.cost);

  const double rel = tolerance<T>();
  EXPECT_NEAR(num.cost, ana.cost, 1e-5 * std::abs(ana.cost));
  EXPECT_NEAR(numerical.computeCost(x), analytical.computeCost(x), 1e-5 * std::abs(ana.cost));
  expectClose(num.jtj, ana.jtj, rel, "JTJ");
  expectClose(num.jtb, ana.jtb, rel, "JTb");
}

}  // namespace

TEST(TestRegressions, InputDimDifferentFromObservationDim) {
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  compareNumericalWithAnalytical<double, PlaneModel<double>>(1000, 2, 1, 4, 0, x);
}

TEST(TestRegressions, InputDimDifferentFromObservationDimFloat) {
  const float x[4]{0.3f, -0.2f, 0.5f, 0.1f};
  compareNumericalWithAnalytical<float, PlaneModel<float>>(1000, 2, 1, 4, 0, x);
}

TEST(TestRegressions, ActiveParamDimLeavesInactiveBlockZero) {
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  PlaneModel<double> model;
  model.active = 2;
  compareNumericalWithAnalytical<double, PlaneModel<double>>(1000, 2, 1, 4, 2, x, model);

  const size_t n = 1000;
  const Data<double> data(n, 2, 1);
  NumericalCostForwardEuler<PlaneModel<double>, double> cost(data.in.data(), data.obs.data(), n, 2, 1, 4, model, 2);
  System<double> system(4);
  cost.computeLinearSystem(x, system.jtj.data(), system.jtb.data(), system.cost);
  for (size_t col = 0; col < 4; ++col) {
    for (size_t row = 0; row < 4; ++row) {
      const double entry = system.jtj[row + 4 * col];
      if (row >= 2 || col >= 2) {
        EXPECT_EQ(entry, 0.0) << "JTJ(" << row << ", " << col << ")";
      } else {
        EXPECT_NE(entry, 0.0) << "JTJ(" << row << ", " << col << ")";
      }
    }
  }
  EXPECT_EQ(system.jtb[2], 0.0);
  EXPECT_EQ(system.jtb[3], 0.0);
  EXPECT_NE(system.jtb[0], 0.0);
}

TEST(TestRegressions, ConfiguredModelIsUsed) {
  const double x[1]{0.3};
  OffsetModel<double> model;
  model.offset = 0.75;
  compareNumericalWithAnalytical<double, OffsetModel<double>>(500, 2, 1, 1, 0, x, model);

  // The cost itself: with the offset it differs from the default model's.
  const size_t n = 500;
  const Data<double> data(n, 2, 1);
  NumericalCostForwardEuler<OffsetModel<double>, double> cost(data.in.data(), data.obs.data(), n, 2, 1, 1, model);
  double expected = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double r = x[0] * data.in[2 * i] + 0.75 - data.obs[i];
    expected += r * r;
  }
  EXPECT_NEAR(cost.computeCost(x), expected, 1e-9 * expected);
}

TEST(TestRegressions, RepeatedCallsAreIdentical) {
  const size_t n = 2000;
  const Data<double> data(n, 2, 1);
  NumericalCostForwardEuler<PlaneModel<double>, double> cost(data.in.data(), data.obs.data(), n, 2, 1, 4);
  const double x[4]{0.3, -0.2, 0.5, 0.1};
  System<double> first(4), second(4);
  cost.computeLinearSystem(x, first.jtj.data(), first.jtb.data(), first.cost);
  cost.computeLinearSystem(x, second.jtj.data(), second.jtb.data(), second.cost);
  EXPECT_EQ(first.cost, second.cost);
  EXPECT_EQ(first.jtj, second.jtj);
  EXPECT_EQ(first.jtb, second.jtb);
}

/// The solution found with numerical jacobians must match the one found with analytical jacobians.
TEST_F(TestTransform2D, TransformLMNumericalMatchesAnalytical) {
  const size_t n = pointcloud_.size();

  Eigen::VectorXd x_analytical{{0, 0, 0}};
  {
    auto solver = std::make_shared<LevenbergMarquardt<double>>(3);
    solver->addCost(std::make_shared<AnalyticalCost<Point2Distance, double>>(
        transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, 2, 2, 3));
    solver->optimize(x_analytical.data());
  }

  Eigen::VectorXd x_numerical{{0, 0, 0}};
  {
    auto solver = std::make_shared<LevenbergMarquardt<double>>(3);
    solver->addCost(std::make_shared<NumericalCostForwardEuler<Point2Distance, double>>(
        transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, 2, 2, 3));
    solver->optimize(x_numerical.data());
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(x_analytical[i], -x0_ref[i], 1e-3) << "analytical, parameter " << i;
    EXPECT_NEAR(x_numerical[i], -x0_ref[i], 1e-3) << "numerical, parameter " << i;
    EXPECT_NEAR(x_numerical[i], x_analytical[i], 1e-4) << "numerical vs analytical, parameter " << i;
  }
}

/// 3D transform of a large cloud: its own suite, so the sizes of the other instantiations of
/// TestTransform3D do not apply.
class TestTransform3DLarge : public TestTransform3D {};

INSTANTIATE_TEST_SUITE_P(OneMillionPoints, TestTransform3DLarge, ::testing::Values(1'000'000));

TEST_P(TestTransform3DLarge, CostMatchesAnalytical) {
  const size_t n = pointcloud_.size();
  NumericalCostForwardEuler<Point3Distance, double> numerical(transformed_pointcloud_[0].data(),
                                                              pointcloud_[0].data(), n, 3, 3, 6);
  AnalyticalCost<Point3Distance, double> analytical(transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, 3, 3,
                                                    6);
  const double x0[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double cost = numerical.computeCost(x0);
  EXPECT_NEAR(cost, analytical.computeCost(x0), 1e-6 * cost);
  // A pure translation by (0.1, 0.1, 0.1): every point is off by 0.03 in squared norm.
  EXPECT_NEAR(cost, 30000.0, 1e-3);
}

TEST_P(TestTransform3DLarge, LinearSystemMatchesAnalytical) {
  const size_t n = pointcloud_.size();
  NumericalCostForwardEuler<Point3Distance, double> numerical(transformed_pointcloud_[0].data(),
                                                              pointcloud_[0].data(), n, 3, 3, 6);
  AnalyticalCost<Point3Distance, double> analytical(transformed_pointcloud_[0].data(), pointcloud_[0].data(), n, 3, 3,
                                                    6);
  const double x0[]{0.1, 0.1, 0.1, 0.0, 0.0, 0.0};

  System<double> num(6), ana(6);
  numerical.computeLinearSystem(x0, num.jtj.data(), num.jtb.data(), num.cost);
  analytical.computeLinearSystem(x0, ana.jtj.data(), ana.jtb.data(), ana.cost);

  EXPECT_NEAR(num.cost, ana.cost, 1e-6 * std::max(1.0, std::abs(ana.cost)));
  expectClose(num.jtj, ana.jtj, 1e-5, "JTJ");
  // The gradient is a sum of terms that nearly cancel at the solution, so compare it to JTJ's scale.
  double jtj_scale = 0.0;
  for (const double v : ana.jtj) jtj_scale = std::max(jtj_scale, std::abs(v));
  for (size_t i = 0; i < 6; ++i) EXPECT_NEAR(num.jtb[i], ana.jtb[i], 1e-5 * jtj_scale) << "JTb index " << i;
}
