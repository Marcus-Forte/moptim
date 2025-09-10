#include <Eigen/Dense>
#include <future>
#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

#include "ConsoleLogger.hh"
#include "Timer.hh"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix dimension>" << std::endl;
    return 1;
  }

  ConsoleLogger logger_;

  int DIM = std::atoi(argv[1]);
  Eigen::MatrixXd A(DIM, DIM);
  A.setRandom();
  Eigen::MatrixXd B(DIM, DIM);
  B.setRandom();
  Eigen::MatrixXd C(DIM, DIM);

  sycl::queue queue{sycl::default_selector_v};
  logger_.log(ILog::Level::DEBUG, "Sycl Device: {}", queue.get_device().get_info<sycl::info::device::name>());

  oneapi::math::backend_selector<oneapi::math::backend::generic> backend_selector(queue);

  Timer t;
  t.start();
  logger_.log(ILog::Level::DEBUG, "CPU -> GPU Copy...");
  double* d_A = sycl::malloc_device<double>(DIM * DIM, queue);
  double* d_B = sycl::malloc_device<double>(DIM * DIM, queue);
  double* d_C = sycl::malloc_device<double>(DIM * DIM, queue);

  queue.copy<double>(A.data(), d_A, DIM * DIM).wait();
  queue.copy<double>(B.data(), d_B, DIM * DIM).wait();
  auto delta_us = t.stop();
  logger_.log(ILog::Level::DEBUG, "Done. Took: {} us", delta_us);

  // Defer GPU to another thread
  auto res = std::async(std::launch::async, [&]() {
    logger_.log(ILog::Level::DEBUG, "GPU Computing...");
    t.start();
    auto res = oneapi::math::blas::generic::column_major::gemm(queue, oneapi::math::transpose::nontrans,
                                                               oneapi::math::transpose::nontrans, DIM, DIM, DIM, 1.0,
                                                               d_A, DIM, d_B, DIM, 0.0, d_C, DIM, {});

    res.wait();
    delta_us = t.stop();
    logger_.log(ILog::Level::DEBUG, "GPU Done. Took: {} us", delta_us);
  });

  logger_.log(ILog::Level::DEBUG, "CPU Computing...");
  t.start();
  C = A * B;
  delta_us = t.stop();
  logger_.log(ILog::Level::DEBUG, "CPU Done. Took: {} us", delta_us);

  res.get();

  Eigen::MatrixXd d_C_copy(DIM, DIM);
  queue.copy(d_C, d_C_copy.data(), DIM * DIM).wait();

  // std::cout << "CPU C = \n" << C << std::endl;
  // std::cout << "GPU C = \n" << d_C_copy << std::endl;
}