#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostForwardEuler.hh"
#include "NumericalCostSycl.hh"
#include "Timer.hh"
#include "test_helper.hh"
#include "transform2d.hh"

const double sycl_vs_cpu_tolerance = 1e-1;

TEST_F(TestTransform2D, SyclCostAndJacobian) {
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};

  auto logger = std::make_shared<ConsoleLogger>();

  const auto num_elements = pointcloud_.size();

  const auto model = std::make_shared<Point2Distance>();

  NumericalCostSycl<double, Point2Distance> num_cost_sycl(logger, queue, transformed_pointcloud_[0].data(),
                                                          pointcloud_[0].data(), 2, 2, 3, num_elements);

  NumericalCostForwardEuler<double> num_cost(transformed_pointcloud_[0].data(), pointcloud_[0].data(), 2, 2, 3,
                                             num_elements, model);

  double x[]{0.0, 0.0, 0.0};

  const auto sycl_cost_result = num_cost_sycl.computeCost(x);
  const auto cost_result = num_cost.computeCost(x);

  EXPECT_NEAR(sycl_cost_result, cost_result, 1e-5);

  // Jacobian
  Timer t0;
  t0.start();

  Eigen::Matrix<double, 3, 3> jtj_sycl;
  Eigen::Matrix<double, 3, 1> jtb_sycl;
  double total_sycl = 0.0;

  Eigen::Matrix<double, 3, 3> jtj;
  Eigen::Matrix<double, 3, 1> jtb;
  double total = 0.0;

  num_cost_sycl.computeLinearSystem(x, jtj_sycl.data(), jtb_sycl.data(), &total_sycl);

  auto stop = t0.stop();
  std::cout << "Sycl cost jacobian: took " << stop << " us" << std::endl;

  t0.start();
  num_cost.computeLinearSystem(x, jtj.data(), jtb.data(), &total);
  stop = t0.stop();
  std::cout << "Known cost jacobian: took " << stop << " us" << std::endl;

  std::cout << "num_jtj_sycl:\n" << jtj_sycl << " " << std::endl;
  std::cout << "num_jtj:\n" << jtj << " " << std::endl;

  EXPECT_NEAR(total_sycl, total, sycl_vs_cpu_tolerance);

  compareMatrices(jtj_sycl, jtj, sycl_vs_cpu_tolerance);
  compareMatrices(jtb_sycl, jtb, sycl_vs_cpu_tolerance);
}

TEST_F(TestTransform2D, Sycl2DTransformLM) {
  auto logger = std::make_shared<ConsoleLogger>();
  const auto num_elements = pointcloud_.size();

  Timer t0;
  t0.start();
  sycl::queue queue{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  auto solver = std::make_shared<LevenbergMarquardt<double>>(3, logger);

  auto cost = std::make_shared<NumericalCostSycl<double, Point2Distance>>(
      logger, queue, transformed_pointcloud_[0].data(), pointcloud_[0].data(), 2, 2, 3, num_elements);                                                         

  double x0[]{0, 0, 0};

  solver->addCost(cost);

  solver->optimize(x0);

  EXPECT_NEAR(x0[0], -x0_ref[0], 1e-3);
  EXPECT_NEAR(x0[1], -x0_ref[1], 1e-3);
  EXPECT_NEAR(x0[2], -x0_ref[2], 1e-3);
  auto delta = t0.stop();
}

// FIXME
// TEST_F(Transform2D, DISABLED_Sycl2DTransformLMAnalytical) {
//   auto logger = std::make_shared<ConsoleLogger>();
//   solver_ = std::make_shared<LevenbergMarquardt>(logger);
//   const auto model = std::make_shared<Point2Distance>();
//   auto cost = std::make_shared<AnalyticalCost>(transformed_pointcloud_[0].data(), pointcloud_[0].data(),
//                                                transformed_pointcloud_.size(), 2, model);

//   solver_->addCost(cost);
//   Eigen::VectorXd x0{{0, 0, 0}};
//   solver_->optimize(x0);

//   EXPECT_NEAR(x0[0], -x0_ref[0], 1e-10);
//   EXPECT_NEAR(x0[1], -x0_ref[1], 1e-10);
//   EXPECT_NEAR(x0[2], -x0_ref[2], 1e-10);
// }
