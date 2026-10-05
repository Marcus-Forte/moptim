#include <gtest/gtest.h>

#include <random>

#include "moptim/AnalyticalCost.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostForwardEuler.hh"

using namespace moptim;

namespace {

// y = a*x + b
struct LineModel {
  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* input, const double* obs, double* res) {
    res[0] = obs[0] - (x[0] * input[0] + x[1]);
  }

  void jacobian(const double* /*x*/, const double* input, const double* /*obs*/, double* jac) {
    jac[0] = -input[0];
    jac[1] = -1.0;
  }
};

struct LineDataset {
  std::vector<double> input;
  std::vector<double> observations;
};

LineDataset makeLineDatasetWithOutliers() {
  LineDataset data;
  constexpr double true_a = 2.0;
  constexpr double true_b = 1.0;
  constexpr int num_inliers = 40;

  for (int i = 0; i < num_inliers; ++i) {
    const double x = static_cast<double>(i) * 0.1;
    data.input.push_back(x);
    data.observations.push_back(true_a * x + true_b);
  }

  // A handful of gross outliers, far from the line.
  for (const double x : {1.0, 2.0, 3.0}) {
    data.input.push_back(x);
    data.observations.push_back(true_a * x + true_b + 50.0);
  }

  return data;
}

}  // namespace

TEST(RobustCost, HuberLossResistsOutliersBetterThanPlainLeastSquares) {
  const LineDataset data = makeLineDatasetWithOutliers();

  auto plain_cost = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  auto robust_cost = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  robust_cost->setLossFunction(std::make_shared<HuberLoss<double>>(1.0));

  LevenbergMarquardt<double> plain_solver(2);
  plain_solver.addCost(plain_cost);
  Eigen::VectorXd x_plain{{0.0, 0.0}};
  plain_solver.optimize(x_plain.data());

  LevenbergMarquardt<double> robust_solver(2);
  robust_solver.addCost(robust_cost);
  Eigen::VectorXd x_robust{{0.0, 0.0}};
  robust_solver.optimize(x_robust.data());

  constexpr double true_a = 2.0;
  constexpr double true_b = 1.0;

  const double plain_error = std::abs(x_plain[0] - true_a) + std::abs(x_plain[1] - true_b);
  const double robust_error = std::abs(x_robust[0] - true_a) + std::abs(x_robust[1] - true_b);

  EXPECT_LT(robust_error, plain_error);
  EXPECT_NEAR(x_robust[0], true_a, 0.05);
  EXPECT_NEAR(x_robust[1], true_b, 0.2);
}

TEST(RobustCost, CauchyLossAlsoResistsOutliers) {
  const LineDataset data = makeLineDatasetWithOutliers();

  auto robust_cost = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  robust_cost->setLossFunction(std::make_shared<CauchyLoss<double>>(1.0));

  LevenbergMarquardt<double> solver(2);
  solver.addCost(robust_cost);
  Eigen::VectorXd x{{0.0, 0.0}};
  solver.optimize(x.data());

  EXPECT_NEAR(x[0], 2.0, 0.1);
  EXPECT_NEAR(x[1], 1.0, 0.3);
}

TEST(RobustCost, ScalarWeightsMatchUniformInformationMatrix) {
  const LineDataset data = makeLineDatasetWithOutliers();
  std::vector<double> weights(data.input.size(), 1.0);
  weights[0] = 4.0;  // trust the first point more

  auto cost_weights = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  cost_weights->setWeights(weights);

  std::vector<Eigen::MatrixXd> information(data.input.size(), Eigen::MatrixXd::Identity(1, 1));
  information[0](0, 0) = 4.0;
  auto cost_information = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  cost_information->setInformation(information);

  Eigen::VectorXd x{{0.5, 0.5}};

  Eigen::MatrixXd jtj_w(2, 2), jtj_i(2, 2);
  Eigen::VectorXd jtb_w(2), jtb_i(2);
  double cost_w = 0.0, cost_i = 0.0;

  cost_weights->computeLinearSystem(x.data(), jtj_w.data(), jtb_w.data(), cost_w);
  cost_information->computeLinearSystem(x.data(), jtj_i.data(), jtb_i.data(), cost_i);

  EXPECT_NEAR(cost_w, cost_i, 1e-10);
  EXPECT_TRUE(jtj_w.isApprox(jtj_i, 1e-10));
  EXPECT_TRUE(jtb_w.isApprox(jtb_i, 1e-10));
}

TEST(RobustCost, WeightsScaleCostQuadratically) {
  const LineDataset data = makeLineDatasetWithOutliers();

  auto unweighted = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);

  std::vector<double> weights(data.input.size(), 2.0);
  auto weighted = std::make_shared<AnalyticalCost<LineModel, double>>(
      data.input.data(), data.observations.data(), data.input.size(), 1, 1, 2);
  weighted->setWeights(weights);

  Eigen::VectorXd x{{0.5, 0.5}};
  const double c_unweighted = unweighted->computeCost(x.data());
  const double c_weighted = weighted->computeCost(x.data());

  EXPECT_NEAR(c_weighted, 2.0 * c_unweighted, 1e-10);
}
