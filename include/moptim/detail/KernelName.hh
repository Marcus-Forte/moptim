#pragma once

// Kernel names that are unique per translation unit.
//
// A SYCL kernel is named after its lambda, which for a lambda in a template mentions the template
// arguments. A model declared in an anonymous namespace mangles to the same string in every
// translation unit (`_GLOBAL__N_1::Model`), so two files that each declare their own `Model` would get
// two kernels with one name, and the device linker silently keeps one of them. The kernels of the
// costs therefore carry `CurrentTuId` in their name: the characters of the compiler's unique stable id
// of a variable with internal linkage, which starts with the per-translation-unit `uid<hash>` prefix
// the driver gives both the host and the device pass (`-fsycl-unique-prefix`).
//
// Limits: two translation units compiled from the same file with the same options get the same
// prefix (the same source built twice into one executable); a model with external linkage defined in
// several translation units gets one kernel per unit, of which the linker keeps the host side of
// one, which is harmless because they are identical.
#include <cstddef>
#include <utility>

namespace moptim::detail {

template <char... Chars>
struct TuIdChars;

constexpr size_t constStringLength(const char* text) {
  size_t length = 0;
  while (text[length] != '\0') {
    ++length;
  }
  return length;
}

// Internal linkage: each translation unit that includes this header gets its own marker, whose unique
// stable id carries that unit's prefix.
namespace {

int tu_marker;

constexpr const char* tuIdString() { return __builtin_sycl_unique_stable_id(tu_marker); }

template <size_t... Is>
auto tuIdType(std::index_sequence<Is...>) -> TuIdChars<tuIdString()[Is]...>;

}  // namespace

/// A type that is different in every translation unit.
using CurrentTuId = decltype(tuIdType(std::make_index_sequence<constStringLength(tuIdString())>{}));

/// Name of the chunk kernel of a cost: evaluation kind, jacobian policy (which names the model), scalar.
template <bool kLinearSystem, class Policy, class T, class TuId>
struct ChunkKernelName;

/// Name of the layout check of a model.
template <class Model, class TuId>
struct ModelLayoutKernelName;

}  // namespace moptim::detail
