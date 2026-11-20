// A second translation unit that instantiates the same kernel template for the same model, to check
// that two units instantiating one kernel link and run. spike_main.cc calls it at startup.
#include <cmath>
#include <cstdio>

#include "spike_kernel.hh"
#include "spike_model.hh"

namespace spike {

bool secondTuSelfCheck() {
  constexpr size_t kElements = 5000;
  const Cloud cloud = makeCloud(kElements);
  const double x[kParamDim] = {0.01, -0.02, 0.03, 0.01, -0.01, 0.02};

  sycl::queue queue{sycl::default_selector_v, {sycl::property::queue::in_order()}};
  LinearSystem<Point3Distance> system(queue, cloud.source, cloud.target, 2, LinearSystem<Point3Distance>::Partitions::Auto);
  Result host, one, two;
  system.runHost(x, host);
  system.runKernel(x, 1, one);
  system.runKernel(x, 2, two);

  const bool same_across_workers = std::memcmp(&one, &two, sizeof(Result)) == 0;
  const bool close_to_host = std::abs(one.cost - host.cost) <= 1e-9 * std::abs(host.cost) && host.cost > 0.0;
  std::fprintf(stderr, "second TU self-check: partitions=%zu W1==W2 bitwise: %s, kernel~host (1e-9): %s\n",
               system.numPartitions(), same_across_workers ? "yes" : "NO", close_to_host ? "yes" : "NO");
  return same_across_workers && close_to_host;
}

}  // namespace spike
