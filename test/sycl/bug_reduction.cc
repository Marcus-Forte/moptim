#include <sycl/sycl.hpp>

constexpr int g_num_elements = 100;

constexpr size_t g_localXSize = 8;  // Row
constexpr size_t g_localYSize = 8;  // Col

constexpr size_t g_param_size = 20;
/// different values for the problem space yield different results.
/// Expected result for 100 elements: 4950

// g_param_size: 8 => reduction result = 4950 OK
// g_param_size: 16 => reduction result = 4050 OK
// g_param_size: 32 => reduction result = 4050 OK

// g_param_size: 22 => reduction result = 4560 NOK
// g_param_size: 20 => reduction result = 3828 NOK (runtime USM error? why?)
// g_param_size: 18 => reduction result = 3828 NOK (runtime USM error? why?)
// g_param_size: 14 => reduction result = 3828 NOK
// g_param_size: 12 => reduction result = 3160 NOK
// g_param_size: 10 => reduction result = 2016 NOK
// g_param_size: 5 => reduction result = 2016 NOK
// g_param_size: 4 => reduction result = 1540 NOK (why? this is power of two)
// g_param_size: 3 => reduction result = 780 NOK
// g_param_size: 3 => reduction result = 780 NOK
// g_param_size: 2 => reduction result = 496 NOK (why? this is power of two)

int main() {
  auto queue = sycl::queue{sycl::default_selector_v};

  std::vector<int> h_elements(g_num_elements);

  std::iota(h_elements.begin(), h_elements.end(), 0);

  const auto h_sum = std::reduce(h_elements.begin(), h_elements.end(), 0, std::plus<>());

  /* Sycl */

  auto* d_elements = sycl::malloc_device<int>(g_num_elements, queue);
  queue.copy<int>(h_elements.data(), d_elements, g_num_elements).wait();

  auto* d_sum = sycl::malloc_device<int>(1, queue);
  auto* d_atomic_sum = sycl::malloc_device<int>(1, queue);

  queue.memset(d_atomic_sum, 0, sizeof(int)).wait();

  queue
      .submit([&](sycl::handler& cgh) {
        auto reduction =
            sycl::reduction<int>(d_sum, 0, sycl::plus<>(), sycl::property::reduction::initialize_to_identity{});

        auto reduction_atomic =
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*d_atomic_sum);

        auto out = sycl::stream(1024, 768, cgh);

        const auto workers = sycl::nd_range<2>({g_num_elements, g_param_size}, {g_localXSize, g_localYSize});

        std::cout << "problem space: " << g_num_elements << "," << g_param_size << std::endl;
        std::cout << "work group space: " << g_localXSize << "," << g_localYSize << std::endl;

        cgh.parallel_for(workers, reduction, [=](sycl::nd_item<2> id, auto& sum) {  // adjust nd_item<2> for 2d range
          const auto itemRow = id.get_global_id(0);
          const auto itemCol = id.get_global_id(1);

          if (itemRow < g_num_elements && itemCol < g_param_size) {
            if (itemCol == 0) {
              sum += d_elements[itemRow];
              reduction_atomic.fetch_add(d_elements[itemRow]);
            }
          }

        });
      })
      .wait();

  int d_sum_result;
  int d_atomic_sum_result;
  queue.copy<int>(d_sum, &d_sum_result, 1).wait();
  queue.copy<int>(d_atomic_sum, &d_atomic_sum_result, 1).wait();

  std::cout << "h Sum (ground truth): " << h_sum << std::endl;
  std::cout << "d reduction Sum: " << d_sum_result << std::endl;
  std::cout << "d Atmoic Add Sum: " << d_atomic_sum_result << std::endl;

  sycl::free(d_elements, queue);
  sycl::free(d_sum, queue);
  sycl::free(d_atomic_sum, queue);
}