# Moptim

Non-linear least-squares library (Gauss-Newton and Levenberg-Marquardt) on Eigen, for the host CPU. The linear
system of every iteration (`J^T J`, `J^T r` and the cost) is assembled by work-item functions over cache-sized
chunks of elements; the partitions of a large cost run in parallel on a oneTBB arena owned by the
`ExecutionContext`, and small costs run inline on the calling thread. The result never depends on the number of
threads. The optimizers and the linear solve run on the calling thread. Parameters can be a plain vector or live
on a manifold, and costs can use analytical or finite-difference Jacobians, per-element weights or information
matrices, and robust losses. (Earlier versions had an optional SYCL backend; it was removed because it was a large
dependency that performed poorly, and host parallelism now comes from oneTBB.)

## Building

moptim is plain C++23 for any compiler (GCC, clang); its public headers are valid C++20. It needs Eigen 3 or 5,
optionally oneTBB (found with `find_package(TBB)`), and, for the tests, GoogleTest (fetched at configure time).

```sh
cmake --preset default      # the image's GCC (/opt/toolchain/gcc.cmake), build/
cmake --build build && ctest --test-dir build
# or, with any other compiler:
cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build <dir> -j && ctest --test-dir <dir>
```

| CMake variable | Default | Meaning |
|---|---|---|
| `MOPTIM_WITH_TBB` | `ON` | run the partitions of large costs on oneTBB. If TBB is not found there is a warning and the build is sequential (`ExecutionContext::parallelAvailable()` is false) |
| `USE_CLANG_TIDY` | `OFF` | run clang-tidy on the library |
| `FORMAT_CODE` | `OFF` | runs clang-format over all `.cc`/`.hh` on build |

oneTBB is used in `src/ExecutionContext.cc` only: no public header includes or names it, and `moptim` links
`TBB::tbb` `PRIVATE` (a static `moptim` passes the link dependency on to its consumers, a consumer needs no TBB
include path). The target is also fine as a subproject: if the parent project found TBB already (its `TBB::tbb`
target exists) it is used as it is, otherwise moptim calls `find_package(TBB)` itself. The `moptim` target passes
nothing but the include path and Eigen on to what links it, and adds no compiler flags beyond `-Wall`.

## Execution context

Work is launched through an `ExecutionContext` (`moptim/ExecutionContext.hh`).

- `ExecutionContext()` uses all hardware threads, `ExecutionContext(size_t threads)` at most `threads`.
  `threads()` / `setThreads()` read and change it (`0` means all hardware threads, `tbb::info::default_concurrency()`);
  `resolvedThreads()` is the number it resolves to (1 without oneTBB).
- **The arena.** With oneTBB a context owns a `tbb::task_arena` whose maximum concurrency is `resolvedThreads()`. It
  is created in the constructor and re-created by `setThreads()`, never lazily, so there is no race on first use.
  A launch of several work-items runs them as tasks of that arena, with the calling thread taking one slot, and
  returns when they are all done. An exception thrown in a worker (a model's `residual`, say) reaches the caller
  of the evaluation. The arena's threads belong to oneTBB's pool; `threads` limits how many of them one context
  uses, not how many exist in the process. Several contexts have independent arenas. Configure a context before
  using it from several threads: `setThreads()` is not synchronized with launches, and a cost does not support
  concurrent evaluations of itself.
- **Launch width.** `launchWidth(partitions, work)` is 1 (inline on the calling thread) below a minimum number of
  partitions, otherwise `min(resolvedThreads(), ceil(partitions / 2))`. The minimum is 6 partitions for a linear
  system and 12 for a cost-only launch (`LaunchWork::LinearSystem` / `CostOnly`; the trial costs of
  Levenberg-Marquardt are cost-only). Waking the arena's workers costs a few microseconds, and a cost-only
  partition is about 15 times less work than a linear-system one, so a small launch is faster inline. The
  minima were measured as the sizes where all threads stop being slower than one on the 28-thread development
  machine (`O = 3`, `P = 6`, double: a partition is 388 elements, so 6 partitions are about 2,300 elements and 12
  about 4,700); a machine with a different wake-up cost may prefer other values
  (`ExecutionContext::kMinPartitionsToGoWide`, `kMinPartitionsToGoWideCostOnly`). `threads == 1` is always inline;
  without oneTBB the width is always 1. The width never changes a result.
- `ExecutionContext::parallelAvailable()` is true when moptim was built with oneTBB.
- **`launches()`**: the number of launches issued so far. An evaluation of a cost is exactly one, whatever its
  width and the number of elements, except for no elements at all (no launch, zeros out). A `FusedCost` evaluation
  is one launch for all its costs.

`defaultContext()` is the process-wide instance (created on first use, with `threads == 0`) and
`setDefaultContext()` replaces it. Every cost constructor takes a trailing
`std::shared_ptr<ExecutionContext> context = nullptr`; null means `defaultContext()`, and a cost keeps the one
it was built with.

**Determinism and threads.** The thread count changes how the work is shared out and never the result. See
[Determinism](#determinism).

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

### Chunked assembly

`AnalyticalCost`, `NumericalCostForwardEuler` and `NumericalCostCentral` are one class template
(`ChunkedCost`) with a Jacobian policy, and there is one assembly: the chunked one, in a single launch per evaluation.

- The chunks are grouped into *partitions* of `max(4, ceil(chunks / 256))` consecutive chunks
  (`detail/Partitioning.hh`), a function of `(N, O, P_active, sizeof(T))` only.
- The work-items of the launch claim partitions through an atomic counter (`std::atomic_ref`, relaxed), run the
  chunk loop over them on their own scratch buffers, and write one partial (cost, `JTb`, lower triangle of `JTJ`)
  per partition.
- The calling thread adds the partials in partition order, mirrors the triangle and zeroes everything outside the
  active block.
- One work-item (width 1) is the plain sequential walk over the partitions, run inline with a plain counter; several
  run on the context's oneTBB arena. A cost that fits in one partition (about 400 elements for `O = 3`, `P = 6`) is
  the plain sequential chunk loop. A numerical cost whose elements fit in one chunk and whose model is
  const-callable steps its one model from evaluation point to evaluation point instead of keeping a copy per point.
- A model that is not const-callable and not copyable can only be used by one work-item, so its cost is always
  inline (width 1), whatever `threads` says.

### Determinism

- The results are **bitwise identical across `threads` settings, core counts and runs**: partitions depend on the
  problem only, never on the threads; each partition's partial is computed by one work-item in a fixed chunk order,
  whichever work-item claims it, and the partials are added in partition order on the calling thread. A parallel
  run therefore writes the same slots and sums them the same way as the inline one. `test/test_determinism.cc`
  compares contexts of 1, 2, 3, 5 and all threads bitwise (costs, linear systems, losses, weights, information
  matrices, manifolds), and the LM results of `bench_moptim` are the same bits at `--threads 1` and `--threads 0`.
- The cost returned with a linear system is **bitwise equal to `computeCost()`** at the same `x` (same chunks,
  partitions and robustification).
- A cost built from plain dimensions gives the same bits as one built from `EuclideanManifold`.
- A plain left-to-right sum over all the elements differs from the partitioned sum in the last bits (they agree to
  about 1e-12 relative in double), unless the cost is a single partition.
- A numerical cost's Jacobian is a difference of nearby residuals and amplifies rounding by about `1/h`: the same
  problem compiled by another compiler, or with another `-ffp-contract`, gives `JTJ` that agree to about 1e-9
  relative, not 1e-12.

### Buffers

A cost allocates its buffers (the models, the copy of `x`, scratch slices, partials) on first use, as aligned heap
memory, and reuses them; steady-state evaluations allocate nothing. Scratch has one slice per work-item and grows
when the launch width grows. The caller's `input` and `observations` arrays are read in place at every evaluation
and must stay valid while the cost uses them. Weights and information factors are copied by `setWeights()` /
`setInformation()`.

## Models

A model is a plain class, nothing is inherited. The cost classes call:

```cpp
struct MyModel {
  void setState(const T* x);                                              // once per evaluation, before any element
  void residual(const T* x, const T* input, const T* obs, T* res) const;  // one element
  void jacobian(const T* x, const T* input, const T* obs, T* jac) const;  // AnalyticalCost only
};
```

- `setState` precomputes whatever depends only on `x` (a rotation matrix, a transform). It runs on the calling
  thread before the launch, so it may do anything (Eigen, heap allocation, trigonometry).
- `jacobian` writes the transposed Jacobian of the element, `active_param_dim × observation_dim` in column-major
  order: `jac[j + active_param_dim * i] = ∂residual[i] / ∂delta[j]`, the derivative with respect to the step
  `delta` at `delta = 0`. For Euclidean parameters that is the ordinary `∂residual[i] / ∂x[j]`; on a manifold
  see [Manifolds](#manifolds).

**Requirements.** There are none beyond ordinary C++: any compiler, any standard library, libm, Eigen, heap
allocation inside `residual` and `jacobian` all work. What matters is threads: when a launch is wide, the work-items
call the model from several threads at once.

- If `residual` and `jacobian` can be called on a `const` model, every work-item uses that one model in place, so
  they must not write to it (no `mutable` state, no shared scratch) and anything it references must be safe to read
  from several threads. Declare them `const` whenever you can.
- Otherwise each work-item copy-constructs its own model (the copy constructor must be safe to run concurrently on
  the same source), uses it and destroys it before the launch ends: one copy per work-item and evaluation (all the
  models of a numerical cost for a linear system, the first for a cost alone).
- A model that is neither const-callable nor copy-constructible is evaluated by one work-item, on the calling thread.
- A model should not throw from `residual`/`jacobian`: an exception does reach the caller of the evaluation, but
  that evaluation's partial sums are lost and the work-items not yet started are skipped. A cost is still usable
  afterwards.

**Where the model lives.** The cost move-constructs it into its own storage on its first evaluation (it must be
move-constructible), calls `setState` on it, destroys it with the cost, and never copies it bytewise.

**Numerical costs keep `1 + P_active` models (forward) or `1 + 2·P_active` (central)**, one per finite-difference
evaluation point (point 0 is `x`), copy-constructed from the model given, once, on the first evaluation that needs
them, with `setState` called on each with its own perturbed `x` before every linear system (before `computeCost`,
only the first). The model must be copy-constructible and copies must not share state that `setState` or `residual`
write to. Memory is `(1 + kP·P_active) · sizeof(Model)`: negligible for a small model, not for one that carries data
(a 24-parameter central cost over a model holding a 24x24 matrix holds 49 copies, about 226 KiB). A numerical cost
whose elements fit in one chunk, with a const-callable model, keeps no copies at all (see
[Chunked assembly](#chunked-assembly)).

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

**Losses are built-in only.** `LossFunction.hh` provides `HuberLoss`, `CauchyLoss`, `GemanMcClureLoss` and
`TrivialLoss`, and `setLossFunction()` throws `std::invalid_argument` for a subclass of your own
(`descriptor().kind == LossKind::Custom`). They are built-in because the work-items evaluate a loss from its
plain-data `LossDescriptor` (`detail::evaluateLoss`, a `switch` on the kind, no virtual call in the hot loop); a
custom `LossFunction::evaluate()` is virtual, and calling it from several worker threads would also make its thread
safety part of the contract. This is a restriction left over from the removed backend, no longer a technical
need: with host threads only, custom losses could come back (a `Custom` descriptor kind that calls the virtual
`evaluate()`, which must then be const and thread-safe), and `LossKind::Custom` is kept for that. The loss object is
shared and its descriptor is read at every evaluation, so a scale changed with `setScale()` between iterations takes
effect on the next.

## Launch overhead and speculative linear systems

Every evaluation is a launch, and Levenberg-Marquardt makes about two per iteration (a linear system and a trial
cost). An inline launch (width 1: a small cost, or `threads == 1`) is a function call. A launch on several
threads pays a hand-off to the arena's workers, a few microseconds plus waiting for the slowest work-item, which is
why a launch goes wide only from 6 partitions (linear system) or 12 (cost only), about 2,300 and 4,700 elements for
`O = 3`, `P = 6`; smaller costs stay inline. What helps for problems of a few hundred to a few thousand elements: put as much as possible in one
cost rather than many small ones (each cost is a launch per evaluation), fuse costs, and turn on speculative linear
systems. The context's `launches()` counter shows what a run costs.

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
of `ChunkedCost`-based costs (`AnalyticalCost`, `NumericalCost*`) that runs their work in one launch: every
work-item works through the partitions of the first cost, then of the second, and so on. Each cost keeps its own
partitions, partials, scratch and model, so its results are those of its own launch, and the fused cost adds them
in the given order. Added to an optimizer instead of its parts it gives the same `x`, iterations and final cost bit
for bit, with one launch per evaluation instead of one per cost; the linear-system cost still equals
`computeCost()`, so speculative linear systems work on it. The launch is as wide as the widest of the costs. The costs
must share their `ExecutionContext` and parameter space (the constructor throws `std::invalid_argument` otherwise),
must not be added to the optimizer as well, and stay usable on their own. A scan registration with a point-to-plane
cost and a one-element IMU factor goes from two launches per evaluation to one; with speculative linear systems a
Levenberg-Marquardt iteration is then one launch.

### Reusing a cost: `setData()` and `setModel()`

An iterative caller that re-associates its data every round (ICP: new correspondences, a model built for the pose of
the round) used to build new costs each round: buffers, the models (one per evaluation point for a numerical
cost), a workspace in the optimizer. `ChunkedCost::setData(input, observations, num_elements)` replaces the elements
(same dimensions) and `setModel(model)` the model, keeping everything else; buffers only grow when the new data
needs more partitions or scratch than before. The results are those of a cost built with the new data and model,
bit for bit. `setData()` removes the per-element noise model (`setWeights()` / `setInformation()`): set it again for
the new elements. The arrays are read at each evaluation, so they must stay valid while the cost uses them. A `FusedCost` sees its costs' new data and models.

## Asynchronous evaluation

`ICost` has `submitLinearSystem(x, JTJ, JTb, &cost)`, `submitCost(x, &cost)` and `wait()` next to the synchronous
`computeLinearSystem()` / `computeCost()`. A submit reads `x` and starts the launch; the outputs are valid after
`wait()`, which also adds the partial results. The defaults call the synchronous functions, so a user subclass of
`ICost` needs no change. A cost has at most one evaluation in flight: any call (compute or submit) on a cost with
one pending completes it first. The optimizers submit every cost before waiting for the first and add the
results in the order of the costs, so they give the same bits as evaluating the costs one after the other.

**Nothing overlaps**: a launch runs inside the submit call (on the calling thread, or on the arena with the calling
thread taking part, and it returns when all work-items are done), and `wait()` only does the host reduction (0.5 to
2.5 microseconds). The API is there so that an `ICost` with truly asynchronous work can use it, and so that the
optimizers' submit-all, wait-all structure is already in place.

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
build/test/bench_moptim                      # everything, all hardware threads
build/test/bench_moptim forward              # only the rows whose name contains "forward"
build/test/bench_moptim --threads 1          # every launch inline on the calling thread
build/test/bench_moptim --threads 4 --speculative lm_optimize
```

- The first line printed is `threads=<resolved> tbb=<on|off>`.
- `--threads N` sets the threads of the default `ExecutionContext` (`0`, the default, is all hardware threads).
- `--speculative` turns on speculative linear systems for every Levenberg-Marquardt row.
- Each row is the best time over several repetitions. Linear-system rows print the cost and the sums of `JTJ` and
  `JTb`; Levenberg-Marquardt rows print the iterations and `launches=` (launches of one `optimize` call).
- Benchmark a Release build, and compare minima over several interleaved runs: a loaded machine is noisy.

`test/compare_bench.py DIR [--ref LABEL] [--tol 1e-12] [--tol-numerical 1e-8]` compares outputs named
`<label>.<round>` (e.g. `t1.1`, `t0.1`): per row, the minimum time per label, the ratio to the reference and the
largest relative difference of every `key=value` note (including `sum(JTJ)`), flagging differences above the
tolerance (the numerical tolerance applies to `forward` and `central` rows). A label without a row prints `-`;
notes the reference does not print are not compared. Between thread counts the notes must agree exactly
(`--tol 0 --tol-numerical 0`); the timings show where the parallel launches pay.

## Consumers (e.g. mslam)

- Link `moptim::moptim` (or add moptim as a subproject). Nothing else is needed from the consumer: the public headers
  are valid C++20 and do not mention oneTBB. Build with `MOPTIM_WITH_TBB=ON` (the default) and TBB installed to get
  parallel launches; without it everything runs on the calling thread with the same results.
- Cost constructors take a trailing `std::shared_ptr<ExecutionContext> context = nullptr` argument; existing calls
  use the default context (all hardware threads). To keep a consumer single-threaded, pass a context with
  `threads == 1` or call `setDefaultContext(std::make_shared<ExecutionContext>(1))` once at start-up.
- Models are called from several threads when a launch is wide: see the [model requirements](#models). Declare
  `residual` and `jacobian` `const` where possible. Custom losses are not accepted (built-in losses only).
- Results do not depend on the thread count. Per evaluation there is one launch: see
  [Launch overhead](#launch-overhead-and-speculative-linear-systems).
