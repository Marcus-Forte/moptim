#pragma once

// How the elements of a cost are split into chunks and the chunks into partitions. Everything here
// is a function of the problem alone (number of elements, observation and active parameter
// dimensions, size of the scalar), never of the number of threads or of the device: the partial sums
// of a linear system are formed per partition and added in partition order, so the result is the same
// whatever number of work-items computed the partitions.
#include <algorithm>
#include <cstddef>

namespace moptim::detail {

/// Budget for the residual and jacobian block of one chunk. It leaves room in a 32 KiB L1 data
/// cache, the smallest in common use, for the chunk's inputs and observations.
inline constexpr size_t kChunkBytes = 16 * 1024;

/// A partition is at least this many chunks, so that claiming one (an atomic increment) and writing
/// its partial sum stay negligible next to the work.
inline constexpr size_t kMinChunksPerPartition = 4;

/// A cost is split in about this many partitions at most: enough to balance the work of any
/// realistic number of cores, few enough that the host sums the partials in a negligible time.
inline constexpr size_t kTargetMaxPartitions = 256;

/// Elements per chunk: as many as fit in kChunkBytes, between 1 and the number of elements.
constexpr size_t chunkElements(size_t num_elements, size_t observation_dim, size_t active_param_dim,
                               size_t value_bytes) {
  const size_t bytes_per_element = observation_dim * (active_param_dim + 1) * value_bytes;
  const size_t fitting = kChunkBytes / std::max<size_t>(bytes_per_element, 1);
  return std::clamp<size_t>(fitting, 1, std::max<size_t>(num_elements, 1));
}

struct Partitioning {
  size_t chunk_elements = 1;
  size_t num_chunks = 0;
  size_t chunks_per_partition = 1;
  size_t num_partitions = 0;
};

/// The chunks and partitions of a problem. Partition `p` covers chunks
/// `[p * chunks_per_partition, min((p + 1) * chunks_per_partition, num_chunks))`.
constexpr Partitioning makePartitioning(size_t num_elements, size_t observation_dim, size_t active_param_dim,
                                        size_t value_bytes) {
  Partitioning result;
  result.chunk_elements = chunkElements(num_elements, observation_dim, active_param_dim, value_bytes);
  result.num_chunks = (num_elements + result.chunk_elements - 1) / result.chunk_elements;
  result.chunks_per_partition = std::max(kMinChunksPerPartition,
                                         (result.num_chunks + kTargetMaxPartitions - 1) / kTargetMaxPartitions);
  result.num_partitions = (result.num_chunks + result.chunks_per_partition - 1) / result.chunks_per_partition;
  return result;
}

}  // namespace moptim::detail
