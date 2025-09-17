
#include <sched.h>

#include <algorithm>

#include "AsyncConsoleLogger.hh"
#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCostForwardEuler.hh"
#include "test_helper.hh"
#include "transform2d.hh"

// One the same CPU, sumulate load with:
//

namespace {
int SetRealTimePriority() {
  struct sched_param sch;
  sch.sched_priority = 99;
  return sched_setscheduler(0, SCHED_FIFO, &sch);
}
void printUsage() {
  std::cout << "Usage: test_realtime <interval_us> <realtime>\n"
            << "  <interval_us> : Loop period in microseconds (e.g., 10000)\n"
            << "  <realtime>    : Set real-time scheduling (1 = enable, 0 = disable)\n"
            << "Example: test_realtime 10000 1\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    printUsage();
    return -1;
  }

  const auto expected_period = std::stoi(argv[1]);
  const auto use_realtime = std::stoi(argv[2]);

  if (use_realtime == 1) {
    if (SetRealTimePriority() != 0) {
      std::cerr << "Unable to set sched to Real time" << std::endl;
      return -1;
    }
  }

  // Prepare
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

  auto logger = std::make_shared<AsyncConsoleLogger>();
  logger->setLevel(ILog::Level::DEBUG);

  std::chrono::high_resolution_clock::time_point last_start =
      std::chrono::high_resolution_clock::now() - std::chrono::microseconds(expected_period);

  auto solver = std::make_shared<LevenbergMarquardt<double>>(3, logger);
  const auto model = std::make_shared<Point2Distance>();

  auto cost = std::make_shared<NumericalCostForwardEuler<double>>(
      transformed_pointcloud[0].data(), pointcloud[0].data(), 2, 2, 3, transformed_pointcloud.size(), model);

  Eigen::VectorXd x0{{0, 0, 0}};

  uint64_t max_latency = 0;

  // This loop has to run exactly every 'interval_us'.
  while (true) {
    const auto start = std::chrono::high_resolution_clock::now();

    solver->clearCosts();
    solver->addCost(cost);
    x0.setZero();
    solver->optimize(x0.data());
    logger->log(ILog::Level::INFO, "Estimated: x={} y={} yaw={}", x0[0], x0[1], x0[2]);

    const uint64_t real_period = std::chrono::duration_cast<std::chrono::microseconds>(start - last_start).count();
    last_start = start;
    const uint64_t compute_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count();
    const auto latency = real_period - expected_period;

    logger->log(ILog::Level::INFO,
                "Intended Period: {}, Real Period: {}, Computation Time: {}, Latency: {}, Max Latency: {} (us)",
                expected_period, real_period, compute_time, latency, max_latency);

    const int64_t sleep_time_us = expected_period - compute_time;

    if (sleep_time_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
      max_latency = std::max(max_latency, latency);
    } else {
      logger->log(ILog::Level::INFO, "Loop period overrun by {} us", -sleep_time_us);
      std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
  }
}