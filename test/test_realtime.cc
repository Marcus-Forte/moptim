
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "Timer.hh"
#include "test_helper.hh"
#include "transform2d.hh"

int main(int argc, char** argv) {
  std::vector<Eigen::Vector2d> pointcloud = read2DTxtScan(TEST_PATH / std::filesystem::path("scan.txt"));
  std::vector<Eigen::Vector2d> transformed_pointcloud;

  Eigen::VectorXd x{{0.1, 0.2, 0.3}};
  Eigen::Rotation2D<double> rot(x[2]);
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();
  transform.translate(Eigen::Vector2d{x[0], x[1]});
  transform.rotate(rot);

  // Transform pointcloud -> transformed_pointcloud by 'x'
  std::transform(pointcloud.begin(), pointcloud.end(), std::back_inserter(transformed_pointcloud),
                 [&](const Eigen::Vector2d& pt) { return transform * pt; });
  //

  auto g_logging = std::make_shared<ConsoleLogger>();
  auto solver = std::make_shared<LevenbergMarquardt>(g_logging);
  Timer t0;
  t0.start();
  const auto model = std::make_shared<Point2Distance>();
  auto cost = std::make_shared<NumericalCost>(transformed_pointcloud[0].data(), pointcloud[0].data(),
                                              transformed_pointcloud.size(), 2, 3, model);
  solver->addCost(cost);
  Eigen::VectorXd x0{{0, 0, 0}};
  solver->optimize(x0);

  auto delta = t0.stop();
}