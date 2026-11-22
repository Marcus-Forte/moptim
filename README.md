# Moptim

Non-linear least-squares library (Gauss-Newton and Levenberg-Marquardt) on Eigen, built on SYCL. The linear
system of every iteration (`J^T J`, `J^T r` and the cost) is assembled in SYCL kernels, one launch per
evaluation, on the CPU (`native_cpu`) or an accelerator; the optimizers and the linear solve run on the calling
thread. Parameters can be a plain vector or live on a manifold, and costs can use analytical or
finite-difference Jacobians, per-element weights or information matrices, and robust losses.

## Building

moptim needs a SYCL compiler: DPC++ (`intel/llvm`) `clang++` with `-fsycl`. The configure step stops with an
error for any other compiler; there is no non-SYCL build. The rest is Eigen 3 and, for the tests, GoogleTest
(fetched at configure time).

```sh
cmake --preset default      # DPC++ in /opt/sycl, Release, Ninja, SYCL target native_cpu, build/
cmake --preset sycl         # native_cpu + nvptx64-nvidia-cuda + spir64, build_sycl/ (needs the CUDA and libspirv bitcode)
cmake --build build && ctest --test-dir build
# or, with the compiler anywhere else:
cmake -S . -B <dir> -G Ninja -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ -DCMAKE_C_COMPILER=<dpcpp>/bin/clang
```

| CMake variable | Default | Meaning |
|---|---|---|
| `SYCL_TARGETS` | `native_cpu` | comma separated `-fsycl-targets` list |
| `MOPTIM_NATIVE_CPU_ARCH` | empty | `-mcpu` value for the `native_cpu` device code, e.g. a core name on AArch64 or an x86-64 level (`x86-64-v3`) on x86 hosts, where it enables AVX in device code. It is applied at link time |
| `MOPTIM_FP_CONTRACT` | `on` | value of `-ffp-contract` for moptim and its consumers. `on` fuses multiply-adds within one expression only, so the result does not depend on the inlining context; `fast` (4 to 19% faster on the analytical costs) let the backend fuse differently per kernel and broke the bitwise cost invariant with Eigen 5. Benchmark reference outputs from builds before this change differ in the last bits for that reason |
| `USE_CLANG_TIDY` | `OFF` | clang-tidy cannot parse a `-fsycl` compilation; configuring with it prints a warning |
| `FORMAT_CODE` | `OFF` | runs clang-format over all `.cc`/`.hh` on build |

The `moptim` target passes its compiler settings on to everything that links it (`PUBLIC`):
`-fsycl -fsycl-targets=...`, `-ffp-contract=...` and the Eigen settings below. The public headers are valid
C++20 (the library itself is C++23).

**Eigen in the device pass.** A `-fsycl` compilation parses every translation unit twice, once for the host
and once per device target. Eigen 3.4's NEON code does not survive the `native_cpu` device front end, so the
build defines `EIGEN_DONT_VECTORIZE` for the device pass only (`-Xsycl-target-frontend=<triple>
-DEIGEN_DONT_VECTORIZE` for each target of `SYCL_TARGETS`), while the host pass keeps Eigen's SIMD code.
`EIGEN_MAX_STATIC_ALIGN_BYTES=16` is defined for both passes so that fixed-size Eigen members (e.g. in a
model) have the same size and alignment on both sides. Eigen in a kernel therefore runs scalar. The library's
own device code uses no Eigen.

For the same reason the `native_cpu` device pass also gets `-U__ARM_NEON -U__ARM_NEON__` (again through
`-Xsycl-target-frontend=native_cpu`, `PUBLIC`, so consumers inherit it): on an aarch64 host the `native_cpu` device
front end crashes on `<arm_neon.h>` ("unexpected Neon vector element type"), which OpenCV's and abseil's (hence
protobuf's) headers include when `__ARM_NEON` is defined. The flags are no-ops on x86 hosts. With a host CPU that
has FP16 arithmetic (e.g. `-mcpu=cortex-a76`) Eigen 5 stores `Eigen::half` as `__fp16`, which the device front end
rejects as a parameter type, so the device pass also gets `-U__ARM_FEATURE_FP16_SCALAR_ARITHMETIC
-U__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`. Including
`<arm_neon.h>` directly in a SYCL translation unit still crashes this compiler; there is no workaround.

## Execution context

Kernels are launched through an `ExecutionContext` (`moptim/ExecutionContext.hh`): an in-order SYCL queue and
the knobs of a launch. `defaultContext()` is the process-wide instance (created on first use) and
`setDefaultContext()` replaces it. Every cost constructor takes a trailing
`std::shared_ptr<ExecutionContext> context = nullptr`; null means `defaultContext()`, and a cost keeps the one
it was built with.

- **`threads`**: the number of work-groups of a launch in the `Chunked` strategy, at most the number of
  partitions of the problem; `0` means all compute units. It decides how the work is shared out and never the
  result. On `native_cpu` a launch of one work-group runs on the calling thread; wider launches use the SYCL
  runtime's thread pool, whose threads exist whatever `threads` says. For literally one thread in the process,
  set `SYCL_NATIVE_CPU_HOST_THREADS=1` in the environment.
- **`AssemblyStrategy`** (`strategy()`, `setStrategy()`): `Chunked`, `WorkGroup` or `Auto` (default), which uses
  `Chunked` on a CPU device and `WorkGroup` on any other. See [Assembly strategies](#assembly-strategies).
- **`workGroupSize()` / `setWorkGroupSize()`**: the work-group length `L` of the `WorkGroup` strategy (0 = 128,
  or the device's maximum if smaller; a cost may use fewer to fit the local memory).
- **`launches()`**: the number of kernel launches issued so far. An evaluation of a cost is exactly one,
  whatever the number of elements, except for no elements at all (no launch, zeros out).

## Key Concepts

### Cost

A `Cost` object encodes the error between model predictions and observations over a dataset. It is constructed with:

- **`input`**: Array of elements (predictions/measurements) that the model function is applied to. Each element has dimensionality `input_dim`. For example, a 2D point cloud would have `input_dim = 2`.
- **`observations`**: Array of measured or observed values corresponding to each input element. Each element has dimensionality `observation_dim`.
- **`input_dim`**: Dimensionality of each input element (e.g. `2` for 2D point clouds).
- **`observation_dim`**: Dimensionality of each observation element (e.g. `2` for 2D measurements).
- **`param_dim`**: Number of entries of the parameter vector `x` — the quantity being optimized, which the model reads. For plain (Euclidean) parameters it is also the size of the linear system. When the parameters live on a [manifold](#manifolds) the cost is built from the manifold instead of `param_dim`: `x` has `manifold->ambientDim()` entries and the linear system `manifold->tangentDim()`.
- **`active_param_dim`** (optional, defaults to `0`): Number of leading *tangent* dimensions (the parameters themselves, if Euclidean) the cost actually depends on; `0` means all of them. The model still receives the full `x`, but only the first `active_param_dim` dimensions are perturbed (numerical costs) or differentiated (analytical costs), and only the corresponding leading block of `JTJ`/`JTb` is populated. Use it when a cost touches a subset of a larger parameter vector (e.g. a scan-matching term using only the pose part of a 24-dim state) to avoid redundant work and avoid writing zero columns. `active_param_dim` must not exceed the tangent dimension.
- **`num_elements`**: Number of input/observation pairs in the dataset.

### Cost Function

The optimizer minimizes the sum of squared residuals over all elements:

```
         N-1
C(x) =   Σ  || f(x, input[i]) - obs[i] ||²  =  r(x)ᵀ · r(x)  =  ||r||²
         i=0

  where:
    x          — parameter vector  (param_dim, the ambient dimension)
    input[i]   — i-th input        (input_dim)
    obs[i]     — i-th observation  (observation_dim)
    f(x, ·)    — model function:   input_dim  →  observation_dim
    r[i]       — residual[i]    =  f(x, input[i]) - obs[i]   (observation_dim)
    r          — full residual  =  [r[0]; r[1]; ...; r[N-1]]  (N*observation_dim)
```

At each optimizer step a linear system is solved for the parameter update `dx`, and the parameters
move to `x ⊞ dx` (`x + dx` for Euclidean parameters, the manifold's `plus` otherwise):

```
  (Jᵀ·J) · dx  =  -Jᵀ · r                     (Gauss-Newton)
  (Jᵀ·J + λ·diag(Jᵀ·J)) · dx  =  -Jᵀ · r      (Levenberg-Marquardt)

  J[i,j]  =  ∂r[i] / ∂dx[j]    Jacobian w.r.t. the step at dx = 0  (N*O × P)
                               (= ∂r[i] / ∂x[j] for Euclidean parameters)
  Jᵀ·J                           Hessian approximation  (P × P)
  Jᵀ·r                           Gradient               (P)
```

Here `P` is the number of dimensions of the step, `tangentDim()`. It is `param_dim` unless the
parameters live on a manifold.

### Dimensions

```
input  (num_elements × input_dim)        observations  (num_elements × observation_dim)
┌─────────────────────────┐              ┌──────────────────────────────────┐
│ elem[0]  x0 x1 ... xI   │              │ elem[0]  y0 y1 ... yO            │
│ elem[1]  x0 x1 ... xI   │              │ elem[1]  y0 y1 ... yO            │
│  ...                    │              │  ...                             │
│ elem[N]  x0 x1 ... xI   │              │ elem[N]  y0 y1 ... yO            │
└─────────────────────────┘              └──────────────────────────────────┘
  N = num_elements, I = input_dim          O = observation_dim

params x  (param_dim)
┌───────────────────────┐
│ p0  p1  p2  ...  pP   │
└───────────────────────┘
  P = param_dim for Euclidean parameters. On a manifold x has ambientDim() entries
  and everything below (J, JTJ, JTb, the step) is sized by tangentDim() instead.
```

```
residual vector r  (N*O)                  Jacobian J  (N*O × P)
┌─────────────────┐                       ┌──────────────────────────┐
│ r[0,0]          │                       │ dr[0,0]/dp0  ...  /dpP   │
│ r[0,1]          │  ← elem 0             │ dr[0,1]/dp0  ...  /dpP   │  ← elem 0
│  ...            │                       │  ...                     │
│ r[1,0]          │                       │ dr[1,0]/dp0  ...  /dpP   │
│ r[1,1]          │  ← elem 1             │ dr[1,1]/dp0  ...  /dpP   │  ← elem 1
│  ...            │                       │  ...                     │
│ r[N,O]          │                       │ dr[N,O]/dp0  ...  /dpP   │
└─────────────────┘                       └──────────────────────────┘
  size: N*O                                 size: (N*O) × P
```

```
JTJ  (P × P)       JTb  (P)
┌───────────┐       ┌────┐
│           │       │    │
│  J^T * J  │       │J^T*r│
│           │       │    │
└───────────┘       └────┘
```

## How a linear system is assembled

The Jacobian above is never stored. A cost walks its elements in chunks small enough to stay in the L1 cache
(16 KiB of residual and Jacobian rows, about 97 elements for `O = 3`, `P = 6` in double precision). For each
chunk it evaluates the residuals and the Jacobian rows into two small buffers and folds them into `JTJ`, `JTb`
and the cost straight away:

```
for each chunk of elements:
    r_chunk  =  residuals of the chunk                (chunk*O)
    J_chunk  =  Jacobian rows of the chunk            (chunk*O × P_active)
    whiten / robustify r_chunk and J_chunk            (only if weights, information or a loss are set)
    JTJ += J_chunkᵀ · J_chunk     JTb += J_chunkᵀ · r_chunk     cost += ||r_chunk||²
```

Memory use and construction time do not grow with the number of elements, which matters when a cost is rebuilt
for every new set of correspondences. A numerical cost runs a chunk through every finite-difference evaluation
point while its inputs are still in cache, writing the residuals at a point straight into its Jacobian column.
The accumulation (`detail/NormalEquationsKernel.hh`) is a fused SIMD kernel for small parameter counts (up to 12
active parameters on AArch64) and a dot-product loop beyond that.

### Assembly strategies

`AnalyticalCost`, `NumericalCostForwardEuler` and `NumericalCostCentral` are one class template
(`ChunkedCost`) with a Jacobian policy, and both strategies run for all of them in a single launch.

- **`Chunked`** (CPUs). The chunks are grouped into *partitions* of `max(4, ceil(chunks / 256))` consecutive
  chunks (`detail/Partitioning.hh`), a function of `(N, O, P_active, sizeof(T))` only. The work-items of the
  launch claim partitions through an atomic counter, run the chunk loop over them on their own scratch buffers,
  and write one partial (cost, `JTb`, lower triangle of `JTJ`) per partition. The host adds the partials in
  partition order, mirrors the triangle and zeroes everything outside the active block. A cost that fits in one
  partition (about 400 elements for `O = 3`, `P = 6`) is the plain sequential chunk loop.
- **`WorkGroup`** (GPUs and other accelerators, `detail/WorkGroupKernel.hh`). One element per work-item; a launch
  has `ceil(N / L)` groups of `L` work-items, group `g` owning the elements `[g·L, min((g+1)·L, N))`. Each
  work-item evaluates its element's residuals and Jacobian rows into work-group local memory and applies the
  noise model and loss; after a barrier the group's work-items compute its sums (cost, `JTb`, lower triangle of
  `JTJ`) entry by entry, each a sequential sum over the group's rows in order. The last group to finish (a
  32-bit atomic counter, release/acquire ordering at device scope) adds the group partials in group order into
  one result slot, so the host reads a single partial. Local memory per element is about `O·(1 + P)` values
  plus a little scratch, and `L` shrinks to fit the device's local memory. A model whose `residual` or
  `jacobian` is not const-callable takes the `Chunked` kernel whatever the strategy.
- **`Auto`** picks `WorkGroup` exactly when `!context.isCpu()`, else `Chunked`.

**GPU performance is unverified.** This repository has no GPU: the `WorkGroup` kernel is tested for
correctness on `native_cpu` (forced with `setStrategy(AssemblyStrategy::WorkGroup)`) and compiles device-only
for `spir64` and `nvptx64-nvidia-cuda`, but it has never run on either, and its `native_cpu` timings say nothing
about a GPU. Known follow-ups once hardware is available: the group partials, counter and result live in shared
USM, which may migrate pages on a discrete GPU (device USM plus one small readback would be better); and the
group sums use one work-item per entry (`K = P(P+1)/2 + P + 1` of the `L` work-items busy), which should get a
GPU-oriented reduction.

### Determinism

- For a given strategy and device the results are **bitwise identical across `threads` settings, core counts and
  runs**: what a partition or a group computes depends only on the problem, and the partials are added in a
  fixed order.
- The cost returned with a linear system is **bitwise equal to `computeCost()`** at the same `x` (same chunks,
  partitions or groups, and robustification).
- A cost built from plain dimensions gives the same bits as one built from `EuclideanManifold`.
- The two strategies sum in different orders, so they agree to rounding (about 1e-12 relative in double), not
  bitwise; an optimizer run can therefore differ in the last bits between strategies, and so can the iteration
  count and status of a run that sits on a convergence threshold. The same holds against a plain left-to-right sum
  over all the elements (the one-partition case).
- A numerical cost's Jacobian is a difference of nearby residuals and amplifies rounding by about `1/h`: the same
  problem compiled by another compiler, or with another `-ffp-contract`, gives `JTJ` that agree to about 1e-9
  relative, not 1e-12.

### Buffers

A cost allocates its USM buffers (the models, the copy of `x`, scratch, partials) on first use and reuses them;
steady-state evaluations allocate nothing and enqueue no copies. The caller's `input` and `observations` arrays
are read in place when the device can read host memory (`native_cpu` can) and are copied to device memory once,
at construction, otherwise. Weights and information factors are stored in USM by `setWeights()` /
`setInformation()`.

## Models

A model is a plain class, nothing is inherited. The cost classes call:

```cpp
struct MyModel {
  void setState(const T* x);                                              // host, once per evaluation, before any element
  void residual(const T* x, const T* input, const T* obs, T* res) const;  // device, one element
  void jacobian(const T* x, const T* input, const T* obs, T* jac) const;  // device, AnalyticalCost only
};
```

- `setState` precomputes whatever depends only on `x` (a rotation matrix, a transform). It runs on the host, so
  it may do anything (Eigen, heap allocation, trigonometry).
- `jacobian` writes the transposed Jacobian of the element, `active_param_dim × observation_dim` in column-major
  order: `jac[j + active_param_dim * i] = ∂residual[i] / ∂delta[j]`, the derivative with respect to the step
  `delta` at `delta = 0`. For Euclidean parameters that is the ordinary `∂residual[i] / ∂x[j]`; on a manifold
  see [Manifolds](#manifolds).

**Device requirements.** `residual` and `jacobian` run in a SYCL kernel: no virtual calls, exceptions, RTTI or
heap allocation (including temporaries of dynamic-size Eigen expressions; fixed-size Eigen is fine and runs
scalar), and no calls to functions defined in another translation unit unless they are `SYCL_EXTERNAL` (or
inline). A reference or pointer to host memory held by the model works only on devices with
`usm_system_allocations` (`native_cpu` has it).

**Standard algorithms.** With GCC >= 16 libstdc++, DPC++ rejects `std::copy`, `std::distance` and the `std::ranges` algorithms on iterators in device code (a false positive: `default_initializable` contains `::new`, reported as "SYCL kernel cannot allocate storage"), and `std::sort` (recursion). Use plain loops or `__builtin_memcpy`; `std::fill`, `std::min`/`std::max`, `std::array` and `std::swap` are fine.

**Math functions.** `std::sin`, `std::exp` and the like work on `native_cpu` (they resolve against the host libm at
link time), but the DPC++ linker prints `Undefined function <name> found in ... This may result in runtime
errors` for each one. `sycl::sin` & co., or the `__SYCL_DEVICE_ONLY__` pattern of `LossFunction.hh`, avoid the
warning and are the portable choice; other devices have no host libm.

**Where the model lives.** The cost move-constructs it into USM on its first evaluation (it must be
move-constructible), calls `setState` on it, destroys it with the cost, and never copies it bytewise. If
`residual` and `jacobian` can be called on a `const` model, every work-item uses that one model in place, so they
must not write to it. Otherwise, in the `Chunked` strategy, each work-item copy-constructs its own model in the
kernel (the copy constructor and destructor then follow the device rules) and destroys it before the launch ends; a
model that is neither const-callable nor copy-constructible is evaluated by one work-item. In the `WorkGroup`
strategy such a model falls back to `Chunked`. Declare `residual` and `jacobian` `const` whenever you can.
`sizeof` and `alignof` of the model must be equal in the host and the device pass (the build gives Eigen the same
alignment in both); the cost checks this once per model type and throws `std::logic_error` otherwise.

**Numerical costs keep `1 + P_active` models (forward) or `1 + 2·P_active` (central)**, one per finite-difference
evaluation point (point 0 is `x`), copy-constructed from the model given, once, on the first evaluation, with
`setState` called on each with its own perturbed `x` before every linear system (before `computeCost`, only the
first). The model must be copy-constructible and copies must not share state that `setState` or `residual`
write to. Memory is `(1 + kP·P_active) · sizeof(Model)` in USM: negligible for a small model, not for one that
carries data (a 24-parameter central cost over a model holding a 24x24 matrix holds 49 copies, about 226 KiB).

**Same-named models in different files.** A SYCL kernel is named after the types it instantiates, and a model in
an anonymous namespace mangles identically in every file. The costs' kernels (and the model layout check)
therefore carry a per-translation-unit id (`detail/KernelName.hh`: the unique stable id of an internal-linkage
variable, which starts with the `uid<hash>` prefix the driver gives each translation unit), so files that each
declare their own `namespace { struct Model }` get their own kernels (`test_tu_models*.cc`). Limits: the same
source file compiled twice with the same options into one executable gets the same prefix, and a kernel cannot
be named after a function-local type, so models must be declared at namespace scope. A model with external
linkage used from several files gets one kernel per file, of which the linker keeps the host side of one; this is
harmless as they are identical.

| Cost | Jacobian from | Model calls per element, per linear system |
|---|---|---|
| `AnalyticalCost<Model, T>` | `Model::jacobian` | 1 residual + 1 jacobian, 1 model |
| `NumericalCostForwardEuler<Model, T>` | forward differences | `1 + P_active` residuals, `1 + P_active` models |
| `NumericalCostCentral<Model, T>` | central differences | `1 + 2·P_active` residuals, `1 + 2·P_active` models |

## Manifolds

Parameters that are not a plain vector (a pose, a rotation) can be optimized directly. The space they live in is
described by `IManifold<T>` (`Manifold.hh`, header only, valid C++20):

```cpp
template <class T>
class IManifold {
 public:
  virtual size_t ambientDim() const = 0;  // entries of x: what costs and models read
  virtual size_t tangentDim() const = 0;  // entries of a step: size of J^T J, J^T b and delta
  // x_plus = x [+] delta. x_plus does not alias x or delta. plus(x, 0) must return x.
  virtual void plus(const T* x, const T* delta, T* x_plus) const = 0;
};
```

moptim ships `EuclideanManifold<T>(dim)` (`plus` is `x + delta`) and `ProductManifold<T>` (a list of
`std::shared_ptr<const IManifold<T>>`; the blocks follow one another in `x` and, in the same order,
in `delta`). It ships no rotation or pose: the storage and the increment convention belong to the
consumer. The interface is a runtime one on purpose, like `ISolver` and `LossFunction`: the
optimizers are explicitly instantiated in `src/`, so a template parameter would force every manifold of a
consumer to be instantiated in the library.

- **Optimizers**: `LevenbergMarquardt<T>(manifold[, solver])` and `GaussNewton<T>(manifold[, solver])`
  take the manifold in place of `dimensions`. The linear system, the step and the solver are sized
  by `tangentDim()`, `x` by `ambientDim()`; both are available as `ambientDim()` / `tangentDim()` of
  the optimizer. The step is applied with `plus`; a rejected Levenberg-Marquardt trial leaves `x`
  untouched. Convergence tests and observer events use the tangent step.
- **Costs**: `AnalyticalCost` and `NumericalCost*` have a constructor taking the manifold where
  `param_dim` goes, and `ICost` gains `tangentDim()` and `manifold()` (null for Euclidean).
  `JTJ` is `tangentDim × tangentDim` and `JTb` has `tangentDim` entries.
- **Same manifold**: an optimizer and its costs are either all built from plain dimensions
  (Euclidean) or all built from the *same manifold object* (`addCost` asserts pointer equality and
  matching dimensions).
- **Jacobian convention**: a model's `jacobian` writes the derivative of the residual with respect
  to the step `delta` at `delta = 0`, in the convention of the manifold's `plus`, with one column
  per tangent dimension. For a pose with `plus(T, delta) = Exp(delta) * T` that is the derivative
  with respect to a left increment; a Jacobian taken for a right increment is silently wrong.
- **Numerical costs** evaluate the residuals at `plus(x, ±h e_p)` for every active tangent
  dimension `p` (computed on the host once per linear system), with the same step `h` as for plain vectors, so
  they give the same quantity as the analytical Jacobian. The models still read the full `x`.
- **`active_param_dim`** counts leading tangent dimensions. With a `ProductManifold` a cost on the
  first block passes that block's `tangentDim()`.

```cpp
// A pose stored as 12 numbers [R | t], tangent [rho, omega], plus(T, d) = Exp(d) * T (consumer code).
class SE3Manifold : public moptim::IManifold<double> { /* 12, 6, plus */ };

const auto pose = std::make_shared<SE3Manifold>();
auto cost = std::make_shared<moptim::AnalyticalCost<PointToPlane, double>>(
    inputs, offsets, num_elements, /*input_dim=*/6, /*observation_dim=*/1, pose);  // PointToPlane::jacobian: 6 values
moptim::LevenbergMarquardt<double> solver(pose);
solver.addCost(cost);
solver.optimize(x);  // x: the 12 numbers of the pose, updated in place and still on the manifold
```

`test/manifold_models.hh` has complete examples (a circle, SE(3), point-to-plane and point-to-point
models) and `test/test_manifold.cc` the checks a consumer manifold should pass.

## Noise model and robust kernels

Every cost can weight its elements and apply a robust kernel (M-estimator):

```cpp
cost->setWeights(weights);          // one scalar weight per element, w = 1/sigma^2
cost->setInformation(information);  // or one observation_dim × observation_dim information matrix per element
cost->setLossFunction(std::make_shared<moptim::HuberLoss<double>>(1.0));
```

With `Ω` the weight or information matrix of an element and `s = rᵀ·Ω·r`, the element contributes `ρ(s)` to the
cost, and its residual and Jacobian rows are whitened and scaled by `sqrt(ρ'(s))` before they are accumulated
(iteratively reweighted least squares). Weights and information matrices replace each other, and an empty vector
removes either.

**Losses are built-in only.** The loss is evaluated in the kernel, so `LossFunction.hh` provides `HuberLoss`,
`CauchyLoss`, `GemanMcClureLoss` and `TrivialLoss`, and `setLossFunction()` throws `std::invalid_argument` for a
subclass of your own (`descriptor().kind == LossKind::Custom`). The loss object is shared and its descriptor is
read at every evaluation, so a scale changed with `setScale()` between iterations takes effect on the next.

## Launch overhead and speculative linear systems

Every evaluation is a launch, about 25 microseconds on `native_cpu`, and Levenberg-Marquardt makes about two per
iteration (a linear system and a trial cost). For a problem of a few hundred to a few thousand elements that is most
of the time (`bench_moptim lm_optimize/curve_fit`: about 30 launches for 13 iterations). What helps: put as much as
possible in one cost rather than many small ones (each cost is a launch per evaluation), and turn on speculative
linear systems. The context's `launches()` counter shows what a run costs.

`LevenbergMarquardt::setSpeculativeLinearSystems(true)` (default off) evaluates each trial point with a full
linear system instead of the cost alone. If the trial is accepted, the next iteration uses that system rather than
assembling one at the new `x`, so an accepted iteration is one launch instead of two. x, the status, the
iteration count and the final cost are identical bit for bit with and without it for the `ChunkedCost`-based costs
(the cost of their linear system is `computeCost()` bitwise).

Use it when most trials are accepted and the launch overhead matters (a handful of iterations on a problem of a few
thousand elements). It loses when many trials are rejected (a rejected trial costs a linear system instead of a
cost) and on large problems where a linear system is much dearer than a cost, and the last accepted iteration of a
run assembles a system that is never used. Do not use it with a custom `ICost` whose `computeLinearSystem()` cost
differs from its `computeCost()`. For observers, `onLinearSystem` still fires once per linear system that is
used: an accepted trial's system is reported at the start of the iteration that uses it, and the systems of
rejected trials are not reported.

### Fusing costs: one launch per evaluation

The launch price is per launch, not per cost, so a problem made of several costs pays it once per cost and
evaluation. `FusedCost<T, Costs...>` (`moptim/FusedCost.hh`, or `makeFusedCost<T>(a, b, ...)`) is an `ICost` made
of `ChunkedCost`-based costs (`AnalyticalCost`, `NumericalCost*`) that runs their work in one kernel: every
work-item works through the partitions of the first cost, then of the second, and so on. Each cost keeps its own
partitions, partials, scratch and model, so its results are those of its own launch, and the fused cost adds them
in the given order. Added to an optimizer instead of its parts it gives the same `x`, iterations and final cost bit
for bit, with one launch per evaluation instead of one per cost; the linear-system cost still equals
`computeCost()`, so speculative linear systems work on it. The costs must share their `ExecutionContext` and
parameter space (the constructor throws `std::invalid_argument` otherwise), must not be added to the optimizer as
well, and stay usable on their own. When the context selects the WorkGroup kernel, the costs are launched one by
one (same results). A scan registration with a point-to-plane cost and a one-element IMU factor goes from two
launches per evaluation to one; with speculative linear systems a Levenberg-Marquardt iteration is then one launch.

### Reusing a cost: `setData()` and `setModel()`

An iterative caller that re-associates its data every round (ICP: new correspondences, a model built for the pose of
the round) used to build new costs each round: buffers in USM, the models (one per evaluation point for a numerical
cost), a workspace in the optimizer. `ChunkedCost::setData(input, observations, num_elements)` replaces the elements
(same dimensions) and `setModel(model)` the model, keeping everything else; buffers only grow when the new data
needs more partitions or scratch than before. The results are those of a cost built with the new data and model,
bit for bit. `setData()` removes the per-element noise model (`setWeights()` / `setInformation()`): set it again for
the new elements. The arrays are read at each evaluation (or copied by `setData()` on a device that cannot read host
memory), so they must stay valid while the cost uses them. A `FusedCost` sees its costs' new data and models.

### The whole solve in one launch: `DeviceLevenbergMarquardt`

Even fused and speculative, Levenberg-Marquardt launches once per iteration, and the host does the damping, the
solve and the step in between. `DeviceLevenbergMarquardt<T, Manifold, Costs...>`
(`moptim/DeviceLevenbergMarquardt.hh`) runs the iterations themselves in a kernel, on one work-item: each cost's
partitions in order with the chunked work-item code, the costs summed in order, a damped Cholesky solve, the step,
the trial point's linear system and the acceptance test, following `LevenbergMarquardt` with speculative linear
systems step for step (same damping, trials, rho, convergence tests and status). `optimize(x)` is one launch. It
ends where `LevenbergMarquardt` does up to rounding (the device pass runs scalar Eigen in the models and its own
Cholesky), with the same iterations and status in the tests. It is opt-in and for small problems, where launches and
not arithmetic dominate; it uses one work-item, so it does not scale with threads, and `LevenbergMarquardt` stays
the reference and the choice for large problems. Requirements: the parameter space is a compile-time
`DeviceManifold` (`kAmbientDim`, `kTangentDim` up to 64, a static device-callable `plus`; `EuclideanSpace<T, N>` for
plain vectors) with the same `plus` as the manifold the costs were built on; the costs are `ChunkedCost`-based, share
one `ExecutionContext`, and their models' `setState()` is device code too (it runs in the kernel); there is no
observer. In mslam's scan registration (about 1000 point-to-plane elements and an IMU factor, 3 iterations per
round) it takes the solve from four launches per round to one, and the replay of `living_room_garden` from 13428
to 3357 launches.

## Asynchronous evaluation

`ICost` has `submitLinearSystem(x, JTJ, JTb, &cost)`, `submitCost(x, &cost)` and `wait()` next to the synchronous
`computeLinearSystem()` / `computeCost()`. A submit reads `x` and starts the launch; the outputs are valid after
`wait()`, which also adds the partial results. The defaults call the synchronous functions, so a user subclass of
`ICost` needs no change. A cost has at most one evaluation in flight: any call (compute or submit) on a cost with
one pending completes it first. The optimizers submit every cost before waiting for the first and add the
results in the order of the costs, so they give the same bits as evaluating the costs one after the other.

**On `native_cpu` this overlaps nothing**: a submit runs the whole kernel inside the call (also with several
work-groups), `wait()` only does the host reduction (0.5 to 2.5 microseconds), and two costs on two queues take as
long submitted together as one after the other. It is there for devices whose queues run asynchronously.

## Observability

The optimizer produces structured telemetry and never formats or writes logs
itself. `optimize()` returns a `Result<T>` (status, iteration count, final cost)
with no logging machinery involved, and an optional `IOptimizerObserver<T>` can
be attached for per-iteration events:

```cpp
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/Observer.hh"

LevenbergMarquardt<double> solver(param_dim);
solver.addCost(cost);

class Recorder : public moptim::IOptimizerObserver<double> {
  void onIteration(const moptim::IterationEvent<double>& e) override { trace.push_back(e); }
  void onLinearSystem(const moptim::LinearSystemEvent<double>& e) override { systems.push_back(e); }
 public:
  std::vector<moptim::IterationEvent<double>> trace;
  std::vector<moptim::LinearSystemEvent<double>> systems;
} recorder;

solver.setObserver(&recorder);
const moptim::Result<double> result = solver.optimize(x);
```

`IterationEvent` carries phase, status, cost/previous cost, `rho`, lambda,
step norm (of the tangent step) and elapsed time. Because it is plain data, benchmarks can record it
in memory (no formatting, allocation or I/O on the hot path) and analyze it
afterwards. With no observer attached, no event payload is gathered and the
clock is not read.

`onLinearSystem` fires once per assembled linear system, which is not always
once per iteration: when Levenberg-Marquardt rejects every trial of an
iteration, `x` has not moved, so the next iteration reuses the system and only
tries larger damping factors.

`Result::status` is `NUMERIC_ERROR` when Gauss-Newton cannot solve its linear
system (`J^T J` not positive definite, or non-finite values); `x` is then left
at the last valid point. Levenberg-Marquardt treats such a system as a rejected
trial and increases the damping.

### Logging is external

moptim ships **no** logging or timing framework. It does not depend on, link,
or include `ILog`, `ConsoleLogger`, `Timer`, Boost, `<iostream>` or `<format>`.
The logging library that used to live in this repository (`utils/`) has moved
out; see `docs/logging.md` for how to bridge events to an external sink.

A consumer with its own logger writes a ~15-line adapter:

```cpp
#include "moptim/Observer.hh"
#include "ILog.hh"   // consumer's logging framework

class LoggingObserver : public moptim::IOptimizerObserver<double> {
 public:
  explicit LoggingObserver(std::shared_ptr<ILog> log) : log_(std::move(log)) {}

  void onIteration(const moptim::IterationEvent<double>& e) override {
    log_->log(ILog::Level::DEBUG, "iter {} status {} cost {} rho {} lambda {} ({} us)",
              e.iteration, static_cast<int>(e.status), e.cost, e.rho, e.lambda, e.elapsed.count() / 1000);
  }
 private:
  std::shared_ptr<ILog> log_;
};
```

## Benchmarks

`bench_moptim` (`test/bench_moptim.cc`) is built with the tests and run by hand. It times `computeLinearSystem()` and
`computeCost()` for every cost class on a 3D point-cloud alignment problem from 1,000 to 1,000,000 elements, the
robust paths, cost construction, complete Levenberg-Marquardt runs and registration loops, the SE(3)
point-to-plane problem through a manifold (`manifold_se3`), and a scan-matcher shape (`two_costs`: a 1200-element
point-to-plane cost plus a one-element numerical prior on the same pose).

```sh
build/test/bench_moptim                      # everything
build/test/bench_moptim forward              # only the rows whose name contains "forward"
build/test/bench_moptim --threads 4 --strategy chunked --speculative lm_optimize
```

- `--threads N` and `--strategy auto|chunked|workgroup` configure the default `ExecutionContext` first
  (`--threads 1` is a single work-group; `--strategy workgroup` is informational on `native_cpu`).
- `--speculative` turns on speculative linear systems for every Levenberg-Marquardt row.
- Each row is the best time over several repetitions. Linear-system rows print the cost and the sums of `JTJ` and
  `JTb`; Levenberg-Marquardt rows print the iterations and `launches=` (kernel launches of one `optimize` call).
- Benchmark a Release build, and compare minima over several interleaved runs: a loaded machine is noisy.

`test/compare_bench.py DIR [--ref LABEL] [--tol 1e-12] [--tol-numerical 1e-8]` compares outputs named
`<label>.<round>` (e.g. `gcc.1`, `t1.1`): per row, the minimum time per label, the ratio to the reference and the
largest relative difference of every `key=value` note (including `sum(JTJ)`), flagging differences above the
tolerance (the numerical tolerance applies to `forward` and `central` rows). A label without a row prints `-`;
notes the reference does not print are not compared.

`bench_launch` (`test/bench_launch.cc`, `bench_launch [latency|pingpong|scaling|threads|phases|workers|alternatives]`)
measures kernel launch latency and multi-core scaling of the default device.

## Migrating a consumer (e.g. mslam)

- Compile with DPC++ (`-fsycl`). Every translation unit that instantiates a cost becomes a SYCL translation unit;
  the `moptim` target passes the flags on to what links it.
- A function called from `residual` or `jacobian` that is defined in another translation unit must be
  `SYCL_EXTERNAL`, or be made inline or header-only. Check the other [model requirements](#models) (no heap
  allocation, virtual calls or exceptions there; `setState` is unrestricted).
- On x86 hosts set `MOPTIM_NATIVE_CPU_ARCH` (e.g. `x86-64-v3`), or the device code is compiled without AVX.
- Cost constructors have a new trailing `std::shared_ptr<ExecutionContext> context = nullptr` argument; existing
  calls compile unchanged and use the default context. Custom losses are no longer accepted (built-in losses only).
- Declare model member functions `const` where possible, and keep model types at namespace scope.
- Results differ from the previous CPU build in the last bits (different summation order, `-ffp-contract`), and so
  may iteration counts at a convergence threshold; they no longer depend on the thread count.
- Per evaluation there is a launch (about 25 µs on `native_cpu`): see
  [Launch overhead](#launch-overhead-and-speculative-linear-systems).
