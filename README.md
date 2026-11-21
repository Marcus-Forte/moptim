# Moptim
Non linear least-squares optimization library on Eigen, built on SYCL.

## Building

moptim is built with a SYCL compiler: DPC++ (`intel/llvm`) `clang++` with `-fsycl` is required, and the
configure step stops with an error for any other compiler. The rest is Eigen 3 and, for the tests,
GoogleTest (fetched at configure time).

```sh
cmake --preset default      # DPC++ in /opt/sycl, Release, Ninja, SYCL target native_cpu, build/
cmake --preset sycl         # native_cpu + nvptx64-nvidia-cuda + spir64, build_sycl/
cmake --build build && ctest --test-dir build
# or, with the compiler anywhere else:
cmake -S . -B <dir> -G Ninja -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ -DCMAKE_C_COMPILER=<dpcpp>/bin/clang
```

| CMake variable | Default | Meaning |
|---|---|---|
| `SYCL_TARGETS` | `native_cpu` | comma separated `-fsycl-targets` list |
| `MOPTIM_NATIVE_CPU_ARCH` | empty | `-mcpu` value for the `native_cpu` device code, e.g. a core name on AArch64 or an x86-64 level on x86 hosts, where it enables AVX in device code. It is applied at link time |
| `MOPTIM_FP_CONTRACT` | `fast` | value of `-ffp-contract` for moptim and its consumers. `fast` (GCC's default) was measured 4 to 19% faster than clang's default `on` on the analytical costs |
| `USE_CLANG_TIDY` | `OFF` | clang-tidy cannot parse a `-fsycl` compilation; configuring with it prints a warning |
| `FORMAT_CODE` | `OFF` | runs clang-format over all `.cc`/`.hh` on build |

The `moptim` target passes its compiler settings on to everything that links it (`PUBLIC`):
`-fsycl -fsycl-targets=...`, `-ffp-contract=fast` (contract `a*b+c` into an FMA across statements, as
GCC does by default) and the Eigen settings below.

**Eigen in the device pass.** A `-fsycl` compilation parses every translation unit twice, once for
the host and once per device target. Eigen 3.4's NEON code does not survive the `native_cpu` device
front end, so the build defines `EIGEN_DONT_VECTORIZE` for the device pass only
(`-Xsycl-target-frontend=<triple> -DEIGEN_DONT_VECTORIZE` for each target of `SYCL_TARGETS`), while
the host pass keeps Eigen's SIMD code. `EIGEN_MAX_STATIC_ALIGN_BYTES=16` is defined for both passes
so that fixed-size Eigen members (e.g. in a model that is copied to the device) have the same size
and alignment on both sides. Code that includes Eigen in a kernel therefore runs scalar Eigen there.

### Execution context

Kernels are launched through an `ExecutionContext` (`moptim/ExecutionContext.hh`): an in-order SYCL
queue, a `threads` knob, an `AssemblyStrategy` and a launch counter. `defaultContext()` is the
process-wide instance and `setDefaultContext()` replaces it. Every cost constructor takes a trailing
`std::shared_ptr<ExecutionContext> context = nullptr`; a null context means `defaultContext()`, and a
cost keeps the one it was built with.

- **`threads`** is the number of work-groups (of one work-item each) of a launch, at most the number of
  partitions of the problem; `0` means all the compute units of the device. It decides how the work is
  shared out and never the result (see [How a linear system is assembled](#how-a-linear-system-is-assembled)).
  On `native_cpu` a launch of one work-group runs on the calling thread; wider launches run on the
  SYCL runtime's thread pool, whose threads exist whatever `threads` says. To have literally one
  thread in the process, also set `SYCL_NATIVE_CPU_HOST_THREADS=1` in the environment.
- **`AssemblyStrategy`**: `Auto` and `Chunked` use the chunked kernel. `WorkGroup` is not implemented
  yet: a cost evaluated with it throws `std::invalid_argument("not implemented yet")`.
- **`launches()`** counts the kernel launches issued through `detail::launch()`: an evaluation of an
  analytical cost is exactly one, whatever the number of elements, except for no elements at all
  (no launch, zeros out). A launch costs about 25 microseconds on `native_cpu`, so a problem of a
  few hundred elements is dominated by it.

All the costs (`AnalyticalCost`, `NumericalCostForwardEuler`, `NumericalCostCentral`) run in kernels. The
optimizers and the linear solve run on the calling thread.

**Asynchronous evaluation.** `ICost` has `submitLinearSystem(x, JTJ, JTb, &cost)`, `submitCost(x, &cost)` and
`wait()` next to the synchronous `computeLinearSystem()` / `computeCost()`. A submit reads `x` and starts the
launch; the outputs are valid after `wait()`, which also adds the partial results. The defaults call the
synchronous functions, so a user subclass of `ICost` needs no change. A cost has at most one evaluation in
flight: any call (compute or submit) on a cost with one pending completes it first. The optimizers submit every
cost before waiting for the first and add the results in the order of the costs, so they give the same bits as
evaluating the costs one after the other. **On `native_cpu` this does not overlap anything**: a submit runs
the whole kernel inside the call (also with several work-groups), and `wait()` only does the host reduction (0.5
to 2.5 microseconds); two costs on two queues take as long submitted together as one after the other. The
interface is there for devices whose queues really run asynchronously.

**Launch overhead.** Every evaluation is a launch, about 25 microseconds on `native_cpu`, and
Levenberg-Marquardt makes about two per iteration (a linear system and a trial cost). For a problem of a
few hundred to a few thousand elements that is most of the time (`bench_moptim lm_optimize/curve_fit` is about
30 launches for 13 iterations). What helps: put as much as possible in one cost rather than many small ones
(each cost is a launch per evaluation), turn on `LevenbergMarquardt::setSpeculativeLinearSystems()` (see
[Speculative linear systems](#speculative-linear-systems)), and prefer analytical to numerical costs only when the
element loop dominates, since the launch count is the same. The context's `launches()` counter shows what a
run costs.

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

### How a linear system is assembled

The Jacobian above is never stored. A cost walks its elements in chunks small enough to stay in
the L1 cache (16 KiB of residual and Jacobian rows, about 97 elements for `O = 3`, `P = 6` in
double precision). For each chunk it evaluates the residuals and the Jacobian rows into two small
buffers and folds them into `JTJ`, `JTb` and the cost straight away:

```
for each chunk of elements:
    r_chunk  =  residuals of the chunk                (chunk*O)
    J_chunk  =  Jacobian rows of the chunk            (chunk*O × P_active)
    whiten / robustify r_chunk and J_chunk            (only if weights, information or a loss are set)
    JTJ += J_chunkᵀ · J_chunk     JTb += J_chunkᵀ · r_chunk     cost += ||r_chunk||²
```

Memory use and construction time therefore do not grow with the number of elements, which matters
when a cost is rebuilt for every new set of correspondences. The numerical costs run a chunk
through every finite-difference evaluation point while its inputs are still in cache, writing the
residuals at a point straight into its Jacobian column, instead of making one pass over the whole
data set per parameter. The accumulation itself
(`detail/NormalEquationsKernel.hh`) is a fused SIMD kernel for small parameter
counts (up to 12 active parameters on AArch64) and a dot-product loop beyond that.

**Partitions.** All the costs (`ChunkedCost`, with an analytical or a numerical Jacobian policy) do the loop above in one kernel launch per
evaluation, on the work-items of the context's queue. The chunks are grouped into *partitions* of
`max(4, ceil(chunks / 256))` consecutive chunks (`detail/Partitioning.hh`), so a partition is a
function of `(N, O, P_active, sizeof(T))` only. Each work-item claims partitions through an atomic
counter, runs the chunk loop over them on its own scratch buffers, and writes one partial result
(cost, `JTb`, lower triangle of `JTJ`) per partition, each in its own cache lines. The host then adds
the partials in partition order, mirrors the lower triangle and zeroes everything outside the active
block. A cost that fits in one partition (up to four chunks, about 400 elements for `O = 3`,
`P = 6`) is the plain sequential chunk loop.

**Determinism.** Because the partials are fixed by the problem and added in a fixed order, the
result is bit-identical for every `threads` setting and every number of cores, run after run.
`computeLinearSystem()` and `computeCost()` use the same chunks, partitions and robustification, so
the cost of a linear system is bit-identical to `computeCost()` at the same `x`. The results are not
bit-identical to a plain left-to-right sum over all the elements (that order is the one-partition
case), only equal to rounding. A numerical cost's Jacobian is a difference of residuals that are
close to each other, so it amplifies rounding by about `1/h`: the same problem compiled by another
compiler (or with another `-ffp-contract`) gives `JTJ` that agree to about 1e-9 relative, not 1e-12.

**Buffers.** A cost allocates its USM buffers (the model, the copy of `x`, the scratch of each
work-item, the partials) on first use and reuses them; steady-state evaluations allocate nothing and
enqueue no copies. The caller's `input` and `observations` arrays are read in place when the device can
read host memory (`native_cpu` can) and are copied to device memory once, at construction, otherwise.
Weights and information factors are stored in USM by `setWeights()` / `setInformation()`.

### Models

A model is a plain class, nothing is inherited. The cost classes call:

```cpp
struct MyModel {
  void setState(const T* x);                                              // host, once per evaluation, before any element
  void residual(const T* x, const T* input, const T* obs, T* res) const;  // device, one element
  void jacobian(const T* x, const T* input, const T* obs, T* jac) const;  // device, AnalyticalCost only
};
```

- `setState` is the place to precompute whatever depends only on `x` (a rotation matrix, a
  transform), so that `residual` and `jacobian` do no per-element setup. It runs on the host, so it
  may do anything (Eigen, heap allocation, trigonometry).
- `jacobian` writes the transposed Jacobian of the element, `active_param_dim × observation_dim` in
  column-major order: `jac[j + active_param_dim * i] = ∂residual[i] / ∂delta[j]`, the derivative with
  respect to the step `delta` at `delta = 0`. For Euclidean parameters that is the ordinary
  `∂residual[i] / ∂x[j]`; on a manifold see [Manifolds](#manifolds).
- **`residual` (and `jacobian`, for `AnalyticalCost`) run in a SYCL kernel**. They may not use virtual
  calls, exceptions, RTTI or heap allocation, which includes the temporaries of dynamic-size Eigen
  expressions (fixed-size Eigen is fine, it runs scalar there), and they may not call functions defined in
  another translation unit unless those are `SYCL_EXTERNAL`. Calls to `std::sin`, `std::exp` and the like work on
  `native_cpu` (they resolve against the host libm when the program is linked) but the DPC++ linker prints
  `Undefined function <name> found in ... This may result in runtime errors` for each one; `sycl::sin` &co.
  (or `LossFunction.hh`'s `__SYCL_DEVICE_ONLY__` pattern) avoid the warning and are the portable choice. A
  reference or pointer to host memory held by the model only works on devices with `usm_system_allocations`
  (`native_cpu` has it).
- **The model lives in USM.** The cost move-constructs it there on its first evaluation (so it must
  be move-constructible), calls `setState` on it, and destroys it with the cost; it is never copied
  bytewise. If `residual` and `jacobian` can be called on a `const` model, every work-item uses that one
  model in place, so they must not write to it. If they cannot, each work-item copy-constructs its own
  model (in the kernel, so the copy constructor and destructor follow the same rules) and destroys it before
  the launch ends; a model that is neither const-callable nor copy-constructible is evaluated by one
  work-item. Declare `residual` and `jacobian` `const` when you can: it avoids the copies.
- `sizeof` and `alignof` of the model must be the same in the host and the device pass. The build gives
  Eigen the same alignment in both (see above), and the cost checks the two sizes once per model type
  and throws `std::logic_error` if they differ.
- **A numerical cost keeps `1 + P_active` models (forward) or `1 + 2·P_active` (central)**, one per
  finite-difference evaluation point (point 0 is `x`). They are copy-constructed from the model given, once,
  on the first evaluation, and `setState` is called on each with its own perturbed `x` before every linear
  system (before every `computeCost`, only the first one is set and used). The model must therefore be
  copy-constructible, and copies must not share state that `setState` or `residual` write to. Memory is
  `(1 + kP·P_active) · sizeof(Model)` in USM, which is negligible for a small model and not for one that
  carries data: a 24-parameter central cost over a model with a 24x24 matrix holds 49 copies of it (about
  226 KiB). With a non-const `residual` each work-item also copies the models it uses, once per launch, into
  its scratch.
- **Kernel names.** A SYCL kernel is named after the types it instantiates, and a model in an anonymous
  namespace mangles the same in every file. The costs' kernels (and the model layout check) therefore carry a
  per-translation-unit id in their name (`detail/KernelName.hh`: the unique stable id of an internal-linkage
  variable, which starts with the `uid<hash>` prefix the driver gives each translation unit), so files that each
  declare their own `namespace { struct Model }` get their own kernels (`test_tu_models*.cc`). Limits: the same source
  file compiled twice with the same options into one executable gets the same prefix; and a model with external
  linkage used from several files gets one kernel per file, of which the linker keeps the host side of one, which
  is harmless as they are identical.

| Cost | Jacobian from | Model calls per element, per linear system |
|---|---|---|
| `AnalyticalCost<Model, T>` | `Model::jacobian` | 1 residual + 1 jacobian, 1 model |
| `NumericalCostForwardEuler<Model, T>` | forward differences | `1 + P_active` residuals, `1 + P_active` models |
| `NumericalCostCentral<Model, T>` | central differences | `1 + 2·P_active` residuals, `1 + 2·P_active` models |

### Manifolds

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

### Noise model and robust kernels

Every cost above can weight its elements and apply a robust kernel (M-estimator):

```cpp
cost->setWeights(weights);          // one scalar weight per element, w = 1/sigma^2
cost->setInformation(information);  // or one observation_dim × observation_dim information matrix per element
cost->setLossFunction(std::make_shared<moptim::HuberLoss<double>>(1.0));
```

With `Ω` the weight or information matrix of an element and `s = rᵀ·Ω·r`, the element contributes
`ρ(s)` to the cost, and its residual and Jacobian rows are whitened and scaled by `sqrt(ρ'(s))`
before they are accumulated (iteratively reweighted least squares). `LossFunction.hh` provides
`HuberLoss`, `CauchyLoss`, `GemanMcClureLoss` and `TrivialLoss`; the kernel is shared, so its scale
can be adapted between iterations with `setScale()` (the analytical cost reads the loss at every
evaluation). Weights and information matrices replace each other, and an empty vector removes either.

`AnalyticalCost` evaluates the loss in its kernel, so it accepts the built-in losses only: for a
`LossFunction` subclass of your own (`descriptor().kind == LossKind::Custom`) `setLossFunction()` throws
`std::invalid_argument`; this holds for the numerical costs too.

### Speculative linear systems

`LevenbergMarquardt::setSpeculativeLinearSystems(true)` (default off) evaluates each trial point with a full
linear system instead of the cost alone. If the trial is accepted, the next iteration uses that system rather
than assembling one at the new `x`, so an accepted iteration is one launch instead of two. x, the status, the
iteration count and the final cost are identical bit for bit with and without it, for the `ChunkedCost`-based
costs (the cost of their linear system is `computeCost()` bitwise).

Use it when most trials are accepted and the launch overhead matters, e.g. a handful of iterations on a
problem of a few thousand elements. It loses when many trials are rejected (a rejected trial costs a linear
system instead of a cost) and on large problems where a linear system is much dearer than a cost, and the
last accepted iteration of a run assembles a system that is never used. Do not use it with a custom `ICost`
whose `computeLinearSystem()` cost differs from its `computeCost()`. `bench_moptim --speculative` runs every
Levenberg-Marquardt row this way; the rows `lm_optimize/two_costs` and `registration_loop/two_costs (25 x 5 it)`
mimic a scan matcher (a 1200-element point-to-plane cost and a one-element numerical prior on SE(3)).

For observers, `onLinearSystem` still fires once per linear system that is used: for an accepted trial's
system it fires at the start of the iteration that uses it (with the time its assembly took), and not at all
for the systems of rejected trials.

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

`bench_moptim` (`test/bench_moptim.cc`) is built with the tests and run by hand. It times
`computeLinearSystem()` and `computeCost()` for every cost class on a 3D point-cloud alignment
problem from 1,000 to 1,000,000 elements, the robust paths, cost construction, and complete
Levenberg-Marquardt runs, plus the SE(3) point-to-plane problem through a manifold (`manifold_se3` rows):

```sh
build/test/bench_moptim            # everything
build/test/bench_moptim forward    # only the rows whose name contains "forward"
build/test/bench_moptim --threads 4 --strategy chunked   # configure the default ExecutionContext first
```

`--speculative` turns on speculative linear systems for every Levenberg-Marquardt row, and the
Levenberg-Marquardt rows print `launches=` (kernel launches of one `optimize` call). `--threads N` and
`--strategy auto|chunked|workgroup` set the default `ExecutionContext` before
anything runs. The analytical rows read it (`--threads 1` is a single work-group; without the option all
compute units are used). `--strategy workgroup` makes every cost row throw.

Each row is the best time over several repetitions. Rows that assemble a linear system also print
the cost and the sums of `JTJ` and `JTb`, so two builds can be compared for numerical equivalence
as well as for speed. Always benchmark a Release build; a standalone configure defaults to it.
