#include <ConsoleLogger.hh>
#include <Eigen/Dense>
#include <iostream>

#include "Timer.hh"

int main(int argc, char** argv) {
  ConsoleLogger logger;

  const auto iterations = 100'000'000;

  auto* data = new double[iterations];

  Timer t0;
  t0.start();

  for (int i = 0; i < iterations; i += 3) {
    Eigen::Map<Eigen::VectorXd> data_map(data + i, 30);
    data_map[0] = data_map.size();
    // data[i] = static_cast<double>(i);
  }

  //   Eigen::Map<Eigen::VectorXd> data_map(data, iterations);
  std::cout << data[6] << std::endl;

  const auto delta = t0.stop(false);
  logger.log(ILog::Level::INFO, "Elapsed time: {} us", delta);

  delete[] data;
}