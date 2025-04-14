#include <gtest/gtest.h>

#include <Timer.hh>

#include "AnalyticalCost.hh"
#include "ConsoleLogger.hh"
#include "NumericalCost2.hh"
#include "test_models.hh"

using namespace test_models;

class SimpleModel2 : public IModel {
 public:
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
  }
  void f(const double* input, const double* measurement, double* f_x) override {
    // Stub implementation
    f_x[0] = measurement[0] - x_[0] * input[0] / (x_[1] + input[0]);
  }

 private:
  double x_[2];
};

TEST(TestCost2, CostEquivalence) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel2>();

  AnalyticalCost<double, double, SimpleModel> an_cost(&x_data_, &y_data_);
  NumericalCost2 num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, model);

  const auto an_cost_result = an_cost.computeCost(x);
  const auto num_cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(an_cost_result, num_cost_result, 1e-5);
}

TEST(TestCost2, JacobianEquivalence2) {
  Eigen::VectorXd x{{0.1, 0.1}};

  const auto model = std::make_shared<SimpleModel2>();

  AnalyticalCost<double, double, SimpleModel> an_cost(&x_data_, &y_data_);
  NumericalCost2 num_cost(x_data_.data(), y_data_.data(), x_data_.size(), 1, model, DifferentiationMethod::CENTRAL);

  const auto [an_jtj, an_jtb, an_total] = an_cost.computeLinearSystem(x);
  const auto [num_jtj, num_jtb, num_total] = num_cost.computeLinearSystem(x);

  for (int i = 0; i < an_jtj.size(); ++i) {
    EXPECT_NEAR(an_jtj(i), num_jtj(i), 1e-5);
  }

  for (int i = 0; i < an_jtb.size(); ++i) {
    EXPECT_NEAR(an_jtb(i), an_jtb(i), 1e-5);
  }

  EXPECT_NEAR(an_total, num_total, 1e-5);

  ConsoleLogger logg;
  logg.log(ILog::Level::INFO, "number: {}", 25);
}