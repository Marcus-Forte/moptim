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
  (Jᵀ·J) · dx  =  Jᵀ · r       (Gauss-Newton)
  (Jᵀ·J + λI) · dx  =  Jᵀ · r  (Levenberg-Marquardt)

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
jacobian_transposed_data_  (P × N*O)      JTJ  (P × P)       JTb  (P)
┌──────────────────────────┐              ┌───────────┐       ┌────┐
│ dr[*]/dp0  ...  dr[*]/dp0│              │           │       │    │
│ dr[*]/dp1  ...  dr[*]/dp1│  = J^T       │  J^T * J  │       │J^T*r│
│  ...                     │              │           │       │    │
│ dr[*]/dpP  ...  dr[*]/dpP│              │           │       │    │
└──────────────────────────┘              └───────────┘       └────┘
  size: P × (N*O)                           size: P × P         size: P
```

## Observability

The optimizer produces structured telemetry and never formats or writes logs
itself. `optimize()` returns a `Result<T>` (status, iteration count, final cost)
with no logging machinery involved, and an optional `IOptimizerObserver<T>` can
be attached for per-iteration events:

```cpp
#include "LevenbergMarquardt.hh"
#include "Observer.hh"

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
afterwards. With no observer attached, no event payload is gathered.

### Logging is external

moptim ships **no** logging or timing framework. It does not depend on, link,
or include `ILog`, `ConsoleLogger`, `Timer`, Boost, `<iostream>` or `<format>`.
The logging library that used to live in this repository (`utils/`) has moved
out; see `docs/logging.md` for how to bridge events to an external sink.

A consumer with its own logger writes a ~15-line adapter:

```cpp
#include "Observer.hh"
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

Build option:

- `MOPTIM_VECTORIZATION_REPORT` — enable clang-only `-Rpass` loop-vectorization
  remarks (default `OFF`).

