# Logging

moptim does **not** contain, build, or depend on any logging or timing
framework. The optimizer emits structured events (`Observer.hh`) and returns a
`Result<T>`; presentation is the consumer's responsibility.

## What was removed

The former in-repo logging framework `utils/` was removed entirely:

```
utils/ILog.hh                utils/ILog.cc
utils/ConsoleLogger.hh       utils/ConsoleLogger.cc
utils/AsyncConsoleLogger.hh  utils/AsyncConsoleLogger.cc
utils/NullLogger.hh
utils/Timer.hh               utils/Timer.cc
utils/CMakeLists.txt
adapters/LoggingObserver.hh  adapters/CMakeLists.txt
```

It should live with the project that owns logging (mslam), preferably as a
namespaced, standalone library, e.g. `mslam::logging` or `mlog`, with its own
repository/build. moptim links only `Eigen3::Eigen`.

## Recovering the code

The files are still in git history. Before the removal is committed, they are
at `HEAD`; afterwards they are at the parent of the removal commit. To restore
them into the logging project:

```sh
# while the removal is uncommitted:
git -C /path/to/moptim archive HEAD utils adapters | tar -x -C /path/to/logging

# after the removal is committed:
git -C /path/to/moptim archive <removed-commit>^ utils adapters | tar -x -C /path/to/logging
```

## Bridging moptim events to a logger

Add this header to the logging library (it depends on both moptim's
`Observer.hh` and the logging framework):

```cpp
#pragma once

#include <memory>
#include <utility>

#include "ILog.hh"        // logging framework
#include "moptim/Observer.hh"    // moptim

namespace moptim {

template <class T>
class LoggingObserver : public IOptimizerObserver<T> {
 public:
  explicit LoggingObserver(std::shared_ptr<ILog> logger) : logger_(std::move(logger)) {}

  void onIteration(const IterationEvent<T>& event) override {
    logger_->log(ILog::Level::DEBUG,
                 "iter {} trial {} phase {} status {} cost {} -> {} rho {} lambda {} delta {} ({} us)",
                 event.iteration, event.trial, static_cast<int>(event.phase), static_cast<int>(event.status),
                 event.cost, event.previous_cost, event.rho, event.lambda, event.delta_norm,
                 event.elapsed.count() / 1000);
  }

  void onLinearSystem(const LinearSystemEvent<T>& event) override {
    logger_->log(ILog::Level::DEBUG, "iter {} linear system cost {} ({} us)", event.iteration, event.cost,
                 event.elapsed.count() / 1000);
  }

 private:
  std::shared_ptr<ILog> logger_;
};

}  // namespace moptim
```

Usage:

```cpp
auto logger = std::make_shared<ConsoleLogger>();
moptim::LoggingObserver<double> observer(logger);
solver.setObserver(&observer);
solver.optimize(x);
```

If the logging framework lives in namespace `mslam::logging`, prefer putting the
adapter there too (`mslam::logging::OptimizerObserver`) so moptim's namespace is
not extended by external code.

## Notes

- A complete, runnable console-logger built directly on `IOptimizerObserver`
  (no logging dependency) is in `test/test_observer.cc`
  (`TestObserver.ConsoleLoggingObserverIllustration`).
- `IOptimizerObserver<T>` is non-owning; keep the observer alive for the
  duration of `optimize()`/`step()`.
- The framework previously named `Timer` is not needed by moptim: timings are
  measured internally with `std::chrono::steady_clock` and delivered in
  `IterationEvent::elapsed` / `LinearSystemEvent::elapsed`, and only when an
  observer is attached.
- Benchmarks should not install a logging observer; record events into memory
  instead.
