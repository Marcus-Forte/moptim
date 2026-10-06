#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"
#include "test_models.hh"

using namespace moptim;
using test_models::TestData;

namespace {

// Residual depends only on the first `K` parameters:
//   r = obs - sum_{j<K} x[j] * input^j
// `jacobian()` writes exactly `NRows` rows of J^T (observation_dim == 1),
// filling rows >= K with zero. This lets the same model family act both as a
// reduced model (NRows == K) and as a full-width model with zero trailing
// columns (NRows > K).
template <class T, int NRows, int K = NRows>
struct PolynomialModel {
  static_assert(K <= NRows);
  void setState(const T* /*x*/) {}

  void residual(const T* x, const T* input, const T* obs, T* res) {
    T prediction = T{0};
    T power = T{1};
    for (int j = 0; j < K; ++j) {
      prediction += x[j] * power;
      power *= input[0];
    }
    res[0] = obs[0] - prediction;
  }

  void jacobian(const T* /*x*/, const T* input, const T* /*obs*/, T* jac) {
    T power = T{1};
    for (int j = 0; j < NRows; ++j) {
      jac[j] = (j < K) ? -power : T{0};
      power *= input[0];
    }
  }
};

constexpr int kFullDim = 5;
constexpr int kActiveDim = 3;

Eigen::VectorXd testX() {
  Eigen::VectorXd x(kFullDim);
  x << 0.2, -0.1, 0.3, 0.7, -0.4;
  return x;
}

void expectZeroTrailing(const Eigen::MatrixXd& jtj, const Eigen::VectorXd& jtb, int active) {
  const int full = static_cast<int>(jtj.rows());
  for (int i = 0; i < full; ++i) {
    for (int j = 0; j < full; ++j) {
      if (i >= active || j >= active) {
        EXPECT_DOUBLE_EQ(jtj(i, j), 0.0);
      }
    }
  }
  for (int i = active; i < full; ++i) EXPECT_DOUBLE_EQ(jtb(i), 0.0);
}

template <class CostA, class CostB>
void expectSameLinearSystem(CostA& a, CostB& b, const double* x, double tol) {
  Eigen::MatrixXd jtj_a(kFullDim, kFullDim), jtj_b(kFullDim, kFullDim);
  Eigen::VectorXd jtb_a(kFullDim), jtb_b(kFullDim);
  double cost_a = 0.0;
  double cost_b = 0.0;
  a.computeLinearSystem(x, jtj_a.data(), jtb_a.data(), cost_a);
  b.computeLinearSystem(x, jtj_b.data(), jtb_b.data(), cost_b);

  EXPECT_NEAR(cost_a, cost_b, tol);
  EXPECT_LT((jtj_a - jtj_b).norm(), tol) << "JTJ mismatch\n" << jtj_a << "\nvs\n" << jtj_b;
  EXPECT_LT((jtb_a - jtb_b).norm(), tol) << "JTb mismatch\n" << jtb_a.transpose() << "\nvs\n" << jtb_b.transpose();
}

template <class Cost>
void configureRobust(Cost& cost, size_t num_elements) {
  std::vector<double> weights(num_elements, 1.0);
  weights[0] = 4.0;
  cost.setWeights(weights);
  cost.setLossFunction(std::make_shared<HuberLoss<double>>(1.0));
}

}  // namespace

// A numerical cost over the full parameter vector whose model only depends on
// the first `kActiveDim` parameters must match the same cost built with
// `active_param_dim = kActiveDim`, for both finite-difference schemes and with
// or without weights / a robust loss.
TEST(ActiveParamDim, ForwardEulerActiveMatchesFullWidth) {
  const Eigen::VectorXd x = testX();
  using ActiveModel = PolynomialModel<double, kActiveDim, kActiveDim>;
  using FullModel = PolynomialModel<double, kFullDim, kActiveDim>;

  for (const bool robust : {false, true}) {
    NumericalCostForwardEuler<ActiveModel, double> active(TestData<double>::x_data_, TestData<double>::y_data_,
                                                          TestData<double>::num_measurements, 1, 1, kFullDim,
                                                          ActiveModel{}, kActiveDim);
    NumericalCostForwardEuler<FullModel, double> full(TestData<double>::x_data_, TestData<double>::y_data_,
                                                      TestData<double>::num_measurements, 1, 1, kFullDim, FullModel{});
    if (robust) {
      configureRobust(active, TestData<double>::num_measurements);
      configureRobust(full, TestData<double>::num_measurements);
    }
    expectSameLinearSystem(active, full, x.data(), 1e-9);

    Eigen::MatrixXd jtj(kFullDim, kFullDim);
    Eigen::VectorXd jtb(kFullDim);
    double cost = 0.0;
    active.computeLinearSystem(x.data(), jtj.data(), jtb.data(), cost);
    expectZeroTrailing(jtj, jtb, kActiveDim);
  }
}

TEST(ActiveParamDim, CentralActiveMatchesFullWidth) {
  const Eigen::VectorXd x = testX();
  using ActiveModel = PolynomialModel<double, kActiveDim, kActiveDim>;
  using FullModel = PolynomialModel<double, kFullDim, kActiveDim>;

  for (const bool robust : {false, true}) {
    NumericalCostCentral<ActiveModel, double> active(TestData<double>::x_data_, TestData<double>::y_data_,
                                                     TestData<double>::num_measurements, 1, 1, kFullDim, ActiveModel{},
                                                     kActiveDim);
    NumericalCostCentral<FullModel, double> full(TestData<double>::x_data_, TestData<double>::y_data_,
                                                 TestData<double>::num_measurements, 1, 1, kFullDim, FullModel{});
    if (robust) {
      configureRobust(active, TestData<double>::num_measurements);
      configureRobust(full, TestData<double>::num_measurements);
    }
    expectSameLinearSystem(active, full, x.data(), 1e-9);

    Eigen::MatrixXd jtj(kFullDim, kFullDim);
    Eigen::VectorXd jtb(kFullDim);
    double cost = 0.0;
    active.computeLinearSystem(x.data(), jtj.data(), jtb.data(), cost);
    expectZeroTrailing(jtj, jtb, kActiveDim);
  }
}

// The analytical cost stores only a `kActiveDim x (obs*N)` jacobian block; its
// assembled linear system must match the full-width analytical cost whose
// trailing columns are explicitly zero.
TEST(ActiveParamDim, AnalyticalActiveMatchesFullWidth) {
  const Eigen::VectorXd x = testX();
  using ActiveModel = PolynomialModel<double, kActiveDim, kActiveDim>;
  using FullModel = PolynomialModel<double, kFullDim, kActiveDim>;

  for (const bool robust : {false, true}) {
    AnalyticalCost<ActiveModel, double> active(TestData<double>::x_data_, TestData<double>::y_data_,
                                               TestData<double>::num_measurements, 1, 1, kFullDim, ActiveModel{},
                                               kActiveDim);
    AnalyticalCost<FullModel, double> full(TestData<double>::x_data_, TestData<double>::y_data_,
                                           TestData<double>::num_measurements, 1, 1, kFullDim, FullModel{});
    if (robust) {
      configureRobust(active, TestData<double>::num_measurements);
      configureRobust(full, TestData<double>::num_measurements);
    }
    expectSameLinearSystem(active, full, x.data(), 1e-10);

    Eigen::MatrixXd jtj(kFullDim, kFullDim);
    Eigen::VectorXd jtb(kFullDim);
    double cost = 0.0;
    active.computeLinearSystem(x.data(), jtj.data(), jtb.data(), cost);
    expectZeroTrailing(jtj, jtb, kActiveDim);
  }
}

// Cross-check the analytical active path against central finite differences.
TEST(ActiveParamDim, AnalyticalActiveMatchesNumerical) {
  const Eigen::VectorXd x = testX();
  using ActiveModel = PolynomialModel<double, kActiveDim, kActiveDim>;

  for (const bool robust : {false, true}) {
    AnalyticalCost<ActiveModel, double> analytical(TestData<double>::x_data_, TestData<double>::y_data_,
                                                   TestData<double>::num_measurements, 1, 1, kFullDim, ActiveModel{},
                                                   kActiveDim);
    NumericalCostCentral<ActiveModel, double> numerical(TestData<double>::x_data_, TestData<double>::y_data_,
                                                        TestData<double>::num_measurements, 1, 1, kFullDim,
                                                        ActiveModel{}, kActiveDim);
    if (robust) {
      configureRobust(analytical, TestData<double>::num_measurements);
      configureRobust(numerical, TestData<double>::num_measurements);
    }
    expectSameLinearSystem(analytical, numerical, x.data(), 1e-6);
  }
}

// computeCost() must ignore the trailing parameters for both cost families.
TEST(ActiveParamDim, ComputeCostIgnoresTrailingParameters) {
  using ActiveModel = PolynomialModel<double, kActiveDim, kActiveDim>;

  Eigen::VectorXd x = testX();
  Eigen::VectorXd x_perturbed = x;
  x_perturbed.tail<kFullDim - kActiveDim>() << 12.0, -7.0;

  AnalyticalCost<ActiveModel, double> analytical(TestData<double>::x_data_, TestData<double>::y_data_,
                                                 TestData<double>::num_measurements, 1, 1, kFullDim, ActiveModel{},
                                                 kActiveDim);
  NumericalCostCentral<ActiveModel, double> numerical(TestData<double>::x_data_, TestData<double>::y_data_,
                                                      TestData<double>::num_measurements, 1, 1, kFullDim, ActiveModel{},
                                                      kActiveDim);

  EXPECT_DOUBLE_EQ(analytical.computeCost(x.data()), analytical.computeCost(x_perturbed.data()));
  EXPECT_DOUBLE_EQ(numerical.computeCost(x.data()), numerical.computeCost(x_perturbed.data()));
}
