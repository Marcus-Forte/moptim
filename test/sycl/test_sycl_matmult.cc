#include <Eigen/Dense>
#include <chrono>
#include <future>
#include <iostream>
#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix dimension>" << std::endl;
    return 1;
  }

  int DIM = std::atoi(argv[1]);
  Eigen::MatrixXd A(DIM, DIM);
  A.setRandom();
  Eigen::MatrixXd B(DIM, DIM);
  B.setRandom();
  Eigen::MatrixXd C(DIM, DIM);

  sycl::queue queue{sycl::default_selector_v};
  std::cout << "Sycl Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  oneapi::math::backend_selector<oneapi::math::backend::generic> backend_selector(queue);

  auto start = std::chrono::steady_clock::now();
  std::cout << "CPU -> GPU Copy..." << std::endl;
  std::span<double> d_A(sycl::malloc_device<double>(DIM * DIM, queue), DIM * DIM);
  std::span<double> d_B(sycl::malloc_device<double>(DIM * DIM, queue), DIM * DIM);
  std::span<double> d_C(sycl::malloc_device<double>(DIM * DIM, queue), DIM * DIM);

  queue.copy<double>(A.data(), d_A.data(), DIM * DIM).wait();
  queue.copy<double>(B.data(), d_B.data(), DIM * DIM).wait();
  auto delta_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Done. Took: " << delta_us << " us" << std::endl;

  // Defer GPU to another thread
  auto res = std::async(std::launch::async, [&]() {
    std::cout << "GPU Computing..." << std::endl;
    start = std::chrono::steady_clock::now();
    auto res = oneapi::math::blas::generic::column_major::gemm(
        queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, DIM, DIM, DIM, 1.0, d_A.data(),
        DIM, d_B.data(), DIM, 0.0, d_C.data(), DIM, {});

    res.wait();
    delta_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "GPU Done. Took: " << delta_us << " us" << std::endl;
  });

  std::cout << "CPU Computing..." << std::endl;
  start = std::chrono::steady_clock::now();
  C = A * B;
  delta_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "CPU Done. Took: " << delta_us << " us" << std::endl;

  res.get();

  Eigen::MatrixXd d_C_copy(DIM, DIM);
  queue.copy(d_C.data(), d_C_copy.data(), DIM * DIM).wait();

  sycl::free(d_A.data(), queue);
  sycl::free(d_B.data(), queue);
  sycl::free(d_C.data(), queue);

  // std::cout << "CPU C = \n" << C << std::endl;
  // std::cout << "GPU C = \n" << d_C_copy << std::endl;
}
