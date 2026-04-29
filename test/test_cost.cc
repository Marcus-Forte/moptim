#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "NumericalCostCentral.hh"
#include "NumericalCostForwardEuler.hh"
#include "test_models.hh"

using namespace test_models;
using namespace moptim;

TEST(TestCost, CostEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  AnalyticalCost<SimpleModel<double>, double> an_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                      TestData<double>::num_measurements, 1, 1, 2,
                                                      SimpleModel<double>{});
  NumericalCostForwardEuler<SimpleModel<double>, double> num_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                                  TestData<double>::num_measurements, 1, 1, 2,
                                                                  SimpleModel<double>{});

  const auto an_cost_result = an_cost.computeCost(x.data());
  const auto num_cost_result = num_cost.computeCost(x.data());

  EXPECT_NEAR(an_cost_result, num_cost_result, 1e-5);
  EXPECT_NEAR(an_cost_result, 0.13670093591408203, 1e-5);
}

TEST(TestCost, JacobianEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  AnalyticalCost<SimpleModel<double>, double> an_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                      TestData<double>::num_measurements, 1, 1, 2,
                                                      SimpleModel<double>{});
  NumericalCostForwardEuler<SimpleModel<double>, double> num_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                                  TestData<double>::num_measurements, 1, 1, 2,
                                                                  SimpleModel<double>{});
  Eigen::MatrixXd num_jtj(2, 2);
  Eigen::VectorXd num_jtb(2);
  double num_total = 0.0;

  Eigen::MatrixXd an_jtj(2, 2);
  Eigen::VectorXd an_jtb(2);
  double an_total = 0.0;

  an_cost.computeLinearSystem(x.data(), an_jtj.data(), an_jtb.data(), an_total);
  num_cost.computeLinearSystem(x.data(), num_jtj.data(), num_jtb.data(), num_total);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(an_jtj(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < an_jtb.size(); ++i) {
    EXPECT_NEAR(an_jtb(i), an_jtb(i), 1e-5);
  }

  EXPECT_NEAR(an_total, num_total, 1e-5);
}

TEST(TestCost, CentralCostEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  NumericalCostForwardEuler<SimpleModel<double>, double> fwd_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                                  TestData<double>::num_measurements, 1, 1, 2);
  NumericalCostCentral<SimpleModel<double>, double> cen_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                             TestData<double>::num_measurements, 1, 1, 2);

  EXPECT_NEAR(fwd_cost.computeCost(x.data()), cen_cost.computeCost(x.data()), 1e-10);
}

TEST(TestCost, CentralJacobianEquivalence) {
  // Central differences are O(h^2) vs forward O(h), so central should be closer to analytical.
  Eigen::VectorXd x{{0.1, 0.1}};

  AnalyticalCost<SimpleModel<double>, double> an_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                      TestData<double>::num_measurements, 1, 1, 2,
                                                      SimpleModel<double>{});
  NumericalCostForwardEuler<SimpleModel<double>, double> fwd_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                                  TestData<double>::num_measurements, 1, 1, 2,
                                                                  SimpleModel<double>{});
  NumericalCostCentral<SimpleModel<double>, double> cen_cost(TestData<double>::x_data_, TestData<double>::y_data_,
                                                             TestData<double>::num_measurements, 1, 1, 2,
                                                             SimpleModel<double>{});

  Eigen::MatrixXd an_jtj(2, 2);
  Eigen::MatrixXd fwd_jtj(2, 2);
  Eigen::MatrixXd cen_jtj(2, 2);
  Eigen::VectorXd an_jtb(2);
  Eigen::VectorXd fwd_jtb(2);
  Eigen::VectorXd cen_jtb(2);
  double an_total = 0.0;
  double fwd_total = 0.0;
  double cen_total = 0.0;

  an_cost.computeLinearSystem(x.data(), an_jtj.data(), an_jtb.data(), an_total);
  fwd_cost.computeLinearSystem(x.data(), fwd_jtj.data(), fwd_jtb.data(), fwd_total);
  cen_cost.computeLinearSystem(x.data(), cen_jtj.data(), cen_jtb.data(), cen_total);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(cen_jtj(i), an_jtj(i), 1e-5);
  }

  // Central differences are O(h^2) vs forward O(h): aggregate error should be smaller.
  const double cen_err = (cen_jtj - an_jtj).norm();
  const double fwd_err = (fwd_jtj - an_jtj).norm();
  EXPECT_LT(cen_err, fwd_err);

  EXPECT_NEAR(cen_total, an_total, 1e-10);
}
