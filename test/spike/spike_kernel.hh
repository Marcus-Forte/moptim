#pragma once
// SYCL side of the spike: the launch of the work-item function, the USM-resident problem, and the
// fixed-order reduction of the partials. Everything is a template or inline so that several
// translation units can instantiate the same kernel.
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

// sycl::khr::launch_grouped is flagged unfinished in this DPC++ release.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

#include "spike_chunk.hh"

namespace spike {

constexpr size_t kInputDim = 3;
constexpr size_t kObservationDim = 3;
constexpr size_t kParamDim = 6;

struct Result {
  double hessian[kMaxParams * kMaxParams] = {};  ///< Full symmetric matrix, column-major, stride `params`.
  double gradient[kMaxParams] = {};
  double cost = 0.0;
};

/// Sums the partials in partition order (independent of how many workers produced them) and
/// mirrors the lower triangle.
template <class Model>
void reducePartials(const Problem<Model>& problem, Result& result) {
  const size_t params = problem.params;
  result = Result{};
  for (size_t partition = 0; partition < problem.num_partitions; ++partition) {
    const double* in = problem.partials + partition * problem.partial_stride;
    for (size_t a = 0; a < params; ++a) {
      for (size_t b = a; b < params; ++b) result.hessian[b + a * params] += *in++;
    }
    for (size_t a = 0; a < params; ++a) result.gradient[a] += *in++;
    result.cost += *in;
  }
  for (size_t a = 0; a < params; ++a) {
    for (size_t b = a + 1; b < params; ++b) result.hessian[a + b * params] = result.hessian[b + a * params];
  }
}

/// One launch of `workers` work-groups of a single work-item each, and one wait. Each work-item
/// claims partitions through an atomic counter in USM (reset by the caller). The kernel reads the
/// problem, and through it the model, by pointer.
template <class Model>
void launchLinearSystem(sycl::queue& queue, const Problem<Model>* problem, size_t workers) {
  sycl::khr::launch_grouped(queue, sycl::range<1>(workers), sycl::range<1>(1), [=](sycl::nd_item<1> item) {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> counter(*problem->counter);
    runWorker(*problem, item.get_group_linear_id(), [&counter] { return counter.fetch_add(size_t{1}); });
  });
  queue.wait();
}

/// Owns the USM allocations of one problem: copies of the data, the model (placement new in shared
/// memory, `setState` on the host), the scratch slices and the partials.
template <class Model>
class LinearSystem {
 public:
  enum class Partitions { Auto, Single };

  LinearSystem(sycl::queue queue, const std::vector<double>& source, const std::vector<double>& target,
               size_t max_workers, Partitions partitions)
      : queue_(std::move(queue)) {
    const size_t n = source.size() / kInputDim;
    input_ = alloc<double>(source.size());
    observations_ = alloc<double>(target.size());
    x_ = alloc<double>(kParamDim);
    std::copy(source.begin(), source.end(), input_);
    std::copy(target.begin(), target.end(), observations_);
    model_ = new (allocBytes(sizeof(Model))) Model();

    Problem<Model>& p = *(problem_ = new (allocBytes(sizeof(Problem<Model>))) Problem<Model>());
    p.model = model_;
    p.input = input_;
    p.observations = observations_;
    p.x = x_;
    p.num_elements = n;
    p.input_dim = kInputDim;
    p.observation_dim = kObservationDim;
    p.params = kParamDim;
    p.chunk_elements = chunkElements(n, kObservationDim, kParamDim);
    const size_t chunks = (n + p.chunk_elements - 1) / p.chunk_elements;
    p.chunks_per_partition = partitions == Partitions::Single ? chunks : std::max<size_t>(4, (chunks + 255) / 256);
    p.num_partitions = (chunks + p.chunks_per_partition - 1) / p.chunks_per_partition;
    const size_t chunk_rows = p.chunk_elements * kObservationDim;
    p.scratch_stride = roundUp8(chunk_rows * (1 + kParamDim) + kParamDim * kObservationDim);
    p.scratch = alloc<double>(p.scratch_stride * max_workers);
    p.partial_stride = roundUp8(packedSize(kParamDim));
    p.partials = alloc<double>(p.partial_stride * p.num_partitions);
    p.counter = alloc<size_t>(1);
    max_workers_ = max_workers;
  }

  LinearSystem(const LinearSystem&) = delete;
  LinearSystem& operator=(const LinearSystem&) = delete;

  ~LinearSystem() {
    model_->~Model();
    for (void* ptr : allocations_) sycl::free(ptr, queue_);
  }

  size_t numPartitions() const { return problem_->num_partitions; }

  /// The work-item function called directly on this thread.
  void runHost(const double* x, Result& result) {
    prepare(x);
    size_t next = 0;
    runWorker(*problem_, 0, [&next] { return next++; });
    reducePartials(*problem_, result);
  }

  /// One launch of `workers` work-groups and a fixed-order reduction.
  void runKernel(const double* x, size_t workers, Result& result) {
    if (workers > max_workers_) throw std::invalid_argument("more workers than scratch slices");
    prepare(x);
    *problem_->counter = 0;
    launchLinearSystem<Model>(queue_, problem_, workers);
    reducePartials(*problem_, result);
  }

 private:
  static size_t roundUp8(size_t n) { return (n + 7) & ~size_t{7}; }

  void* allocBytes(size_t bytes) {
    void* ptr = sycl::aligned_alloc_shared(64, bytes, queue_);
    if (ptr == nullptr) throw std::bad_alloc();
    allocations_.push_back(ptr);
    return ptr;
  }
  template <class U>
  U* alloc(size_t count) {
    return static_cast<U*>(allocBytes(std::max<size_t>(count, 1) * sizeof(U)));
  }

  void prepare(const double* x) {
    std::copy(x, x + kParamDim, x_);
    model_->setState(x_);
  }

  sycl::queue queue_;
  std::vector<void*> allocations_;
  double* input_ = nullptr;
  double* observations_ = nullptr;
  double* x_ = nullptr;
  Model* model_ = nullptr;
  Problem<Model>* problem_ = nullptr;
  size_t max_workers_ = 0;
};

}  // namespace spike
