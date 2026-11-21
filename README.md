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
queue, a `threads` knob (the number of work-groups of a launch, `0` = all compute units; it never
changes a result), an `AssemblyStrategy` and a launch counter. `defaultContext()` is the process-wide
instance, `setDefaultContext()` replaces it. The costs still assemble their linear systems on the
host; the context and the device kernels in `moptim/detail/` (`NormalEquationsKernel.hh`, `Usm.hh`)
are the infrastructure they will move onto.

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
through every finite-difference evaluation point while its inputs are still in cache, instead of
making one pass over the whole data set per parameter. The accumulation itself
(`accumulateNormalEquations`, `src/NormalEquations.cc`) is a fused SIMD kernel for small parameter
counts (up to 12 active parameters on AArch64) and a dot-product loop beyond that.

### Models

A model is a plain class, nothing is inherited. The cost classes call:

```cpp
struct MyModel {
  void setState(const T* x);                                        // once per evaluation, before any element
  void residual(const T* x, const T* input, const T* obs, T* res);  // one element
  void jacobian(const T* x, const T* input, const T* obs, T* jac);  // AnalyticalCost only
};
```

- `setState` is the place to precompute whatever depends only on `x` (a rotation matrix, a
  transform), so that `residual` and `jacobian` do no per-element setup.
- `jacobian` writes the transposed Jacobian of the element, `active_param_dim × observation_dim` in
  column-major order: `jac[j + active_param_dim * i] = ∂residual[i] / ∂delta[j]`, the derivative with
  respect to the step `delta` at `delta = 0`. For Euclidean parameters that is the ordinary
  `∂residual[i] / ∂x[j]`; on a manifold see [Manifolds](#manifolds).
- When its elements span more than one chunk, a numerical cost keeps one copy of the model per
  finite-difference evaluation point and calls `setState` on each copy with its own perturbed `x`.
  The model must therefore be copy-constructible, and copies must not share state that `setState`
  or `residual` write to. A cost that fits in a single chunk uses its one model for every point.

| Cost | Jacobian from | Model calls per element, per linear system |
|---|---|---|
| `AnalyticalCost<Model, T>` | `Model::jacobian` | 1 residual + 1 jacobian |
| `NumericalCostForwardEuler<Model, T>` | forward differences | `1 + P_active` residuals |
| `NumericalCostCentral<Model, T>` | central differences | `1 + 2·P_active` residuals |

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
  dimension `p` (computed once per linear system), with the same step `h` as for plain vectors, so
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
can be adapted between iterations with `setScale()`. Weights and information matrices replace each
other, and an empty vector removes either.

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

`--threads N` and `--strategy auto|chunked|workgroup` set the default `ExecutionContext` before
anything runs; the costs do not read it yet, so they do not change the timings.

Each row is the best time over several repetitions. Rows that assemble a linear system also print
the cost and the sums of `JTJ` and `JTb`, so two builds can be compared for numerical equivalence
as well as for speed. Always benchmark a Release build; a standalone configure defaults to it.
