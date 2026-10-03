# Moptim

C++23 non-linear least-squares library (Gauss-Newton, Levenberg-Marquardt) on Eigen, with optional SYCL. See `readme.md` for the math and dimension conventions.

## Build / test
- Configure with presets: `cmake --preset default` (Ninja, Release, binary dir `build/`, uses toolchain `/opt/toolchain/gcc.cmake`) or `--preset sycl` (`build_sycl/`, needs `/opt/sycl` clang and oneMath; may be unavailable in this container).
- Build: `cmake --build build`. Tests: `ctest --test-dir build` (gtest, fetched by FetchContent at configure time, so it needs network). Test binary is `test_moptim`; run one test with `build/test/test_moptim --gtest_filter=Suite.Name`.
- Only `-Wall` is set (no `-Werror`). `build/` and `build_sycl/` are gitignored.
- Optional flags: `-DUSE_CLANG_TIDY=ON` (not compatible with SYCL), `-DFORMAT_CODE=ON` (runs clang-format -i over all `.cc`/`.hh` on build).

## Layout
- `include/moptim/*.hh`: mostly header templates (costs, models, observer, result). Consumers include them as `"moptim/<Header>.hh"`, since `${CMAKE_CURRENT_SOURCE_DIR}/include` is the public include root. `src/` has only `EigenSolver.cc`, `GaussNewton.cc` and `LevenbergMarquardt.cc`, which use explicit instantiation. Supporting a new scalar type means editing `src/*.cc`.
- `test/`: gtest suites. New test files must be added to `test/CMakeLists.txt`. `test/nist/` holds the NIST StRD regression datasets, and `test/sycl/` is built only with `WITH_SYCL`.
- Library has no logging or timing. Telemetry goes through `IOptimizerObserver` (`Observer.hh`). Do not add `<iostream>`, `<format>` or logger dependencies (see `docs/logging.md`).
- `plan.md` and `improvements.md` are notes and reviews, not specs. They may be stale (`improvements.md` mentions removed `utils/`).
