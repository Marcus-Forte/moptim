
#include <sched.h>

#include "ConsoleLogger.hh"
#include "LevenbergMarquardt.hh"
#include "NumericalCost.hh"
#include "test_helper.hh"
#include "transform2d.hh"

namespace {
int SetRealTimePriority() {
  struct sched_param sch;
  sch.sched_priority = 99;
  return sched_setscheduler(0, SCHED_FIFO, &sch);
}

void printUsage() { std::cout << "Usage: test_realtime [ interval us ] [ realtime 1|0] " << std::endl; }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    printUsage();
    return -1;
  }

  const auto interval_us = std::stoi(argv[1]);
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

  auto g_logging = std::make_shared<ConsoleLogger>();
  g_logging->setLevel(ILog::Level::INFO);

  // This loop has to run exactly every 'interval_us'.
  std::chrono::high_resolution_clock::time_point last_start = {};

  while (true) {
    auto solver = std::make_shared<LevenbergMarquardt>(g_logging);

    const auto start = std::chrono::high_resolution_clock::now();

    const auto model = std::make_shared<Point2Distance>();
    auto cost = std::make_shared<NumericalCost>(transformed_pointcloud[0].data(), pointcloud[0].data(),
                                                transformed_pointcloud.size(), 2, 3, model);
    solver->addCost(cost);
    Eigen::VectorXd x0{{0, 0, 0}};
    solver->optimize(x0);
    g_logging->log(ILog::Level::INFO, "Estimated: x={} y={} yaw={}", x0[0], x0[1], x0[2]);

    const auto end = std::chrono::high_resolution_clock::now();

    const auto delta = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const int64_t sleep_time_us = interval_us - delta;

    if (sleep_time_us > 0) {
      g_logging->log(ILog::Level::INFO, "Computation time: {} us. Sleeping for {} us", delta, sleep_time_us);
      // std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
      // replace above with clock_nanosleep for better accuracy
      timespec ts = {0, sleep_time_us * 1000};  // 100,000,000 nanoseconds = 0.1 seconds

      clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);

    } else {
      g_logging->log(ILog::Level::INFO, "Computation time: {} us. Overrun by {} us", delta, -sleep_time_us);
      std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }

    const auto real_period = std::chrono::duration_cast<std::chrono::microseconds>(start - last_start).count();
    g_logging->log(ILog::Level::INFO, "Intended Period: {}, Real Period: {} us. Latency: {} us", interval_us,
                   real_period, real_period - interval_us);
    last_start = start;
  }
}