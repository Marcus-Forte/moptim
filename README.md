# Moptim
Non linear optimization library built with option for SYCL.

## Key Concepts

### Cost

A `Cost` object encodes the error between model predictions and observations over a dataset. It is constructed with:

- **`input`**: Array of elements (predictions/measurements) that the model function is applied to. Each element has dimensionality `input_dim`. For example, a 2D point cloud would have `input_dim = 2`.
- **`observations`**: Array of measured or observed values corresponding to each input element. Each element has dimensionality `observation_dim`.
- **`input_dim`**: Dimensionality of each input element (e.g. `2` for 2D point clouds).
- **`observation_dim`**: Dimensionality of each observation element (e.g. `2` for 2D measurements).
- **`param_dim`**: Dimensionality of the parameter vector — the quantity being optimized. The optimizer iteratively updates this vector to minimize the residuals between model predictions and observations.
- **`active_param_dim`** (optional, defaults to `0`): Number of leading parameters the cost actually depends on; `0` means the full `param_dim`. The model still receives the full parameter vector, but only the first `active_param_dim` entries are perturbed (numerical costs) or differentiated (analytical costs), and only the corresponding leading block of `JTJ`/`JTb` is populated. Use it when a cost touches a subset of a larger parameter vector (e.g. a scan-matching term using only the pose part of a 24-dim state) to avoid redundant work and avoid writing zero columns. `active_param_dim` must not exceed `param_dim`.
- **`num_elements`**: Number of input/observation pairs in the dataset.

### Cost Function

The optimizer minimizes the sum of squared residuals over all elements:

```
         N-1
C(x) =   Σ  || f(x, input[i]) - obs[i] ||²  =  r(x)ᵀ · r(x)  =  ||r||²
         i=0

  where:
    x          — parameter vector  (param_dim)
    input[i]   — i-th input        (input_dim)
    obs[i]     — i-th observation  (observation_dim)
    f(x, ·)    — model function:   input_dim  →  observation_dim
    r[i]       — residual[i]    =  f(x, input[i]) - obs[i]   (observation_dim)
    r          — full residual  =  [r[0]; r[1]; ...; r[N-1]]  (N*observation_dim)
```

At each optimizer step a linear system is solved for the parameter update `dx`:

```
  (Jᵀ·J) · dx  =  -Jᵀ · r                     (Gauss-Newton)
  (Jᵀ·J + λ·diag(Jᵀ·J)) · dx  =  -Jᵀ · r      (Levenberg-Marquardt)

  J[i,j]  =  ∂r[i] / ∂x[j]     Jacobian  (N*O × P)
  Jᵀ·J                           Hessian approximation  (P × P)
  Jᵀ·r                           Gradient               (P)
```

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
  P = param_dim
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
  column-major order: `jac[j + active_param_dim * i] = ∂residual[i] / ∂x[j]`.
- When its elements span more than one chunk, a numerical cost keeps one copy of the model per
  finite-difference evaluation point and calls `setState` on each copy with its own perturbed `x`.
  The model must therefore be copy-constructible, and copies must not share state that `setState`
  or `residual` write to. A cost that fits in a single chunk uses its one model for every point.

| Cost | Jacobian from | Model calls per element, per linear system |
|---|---|---|
| `AnalyticalCost<Model, T>` | `Model::jacobian` | 1 residual + 1 jacobian |
| `NumericalCostForwardEuler<Model, T>` | forward differences | `1 + P_active` residuals |
| `NumericalCostCentral<Model, T>` | central differences | `1 + 2·P_active` residuals |

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
step norm and elapsed time. Because it is plain data, benchmarks can record it
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
Levenberg-Marquardt runs:

```sh
build/test/bench_moptim            # everything
build/test/bench_moptim forward    # only the rows whose name contains "forward"
```

Each row is the best time over several repetitions. Rows that assemble a linear system also print
the cost and the sums of `JTJ` and `JTb`, so two builds can be compared for numerical equivalence
as well as for speed. Always benchmark a Release build; a standalone configure defaults to it.
