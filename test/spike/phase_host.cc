// Host-only (no SYCL) timing of the chunk loop pieces, to find where the spike loses time against
// the library. Build by hand: clang++ -O3 -std=c++20 -I/usr/include/eigen3 phase_host.cc
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "spike_chunk.hh"
#include "spike_model.hh"
using namespace spike;
using Clock = std::chrono::steady_clock;
template <class F> double best(F&& f) {
  f();
  double b = 1e30;
  for (int r = 0; r < 7; ++r) {
    size_t calls = 0; auto s = Clock::now(); double el;
    do { f(); ++calls; el = std::chrono::duration<double, std::nano>(Clock::now() - s).count(); } while (el < 40e6);
    b = std::min(b, el / calls);
  }
  return b;
}
volatile double sink;
int main() {
  const size_t n = 1000000;
  Cloud c = makeCloud(n);
  const double x[6] = {0.01, -0.02, 0.03, 0.01, -0.01, 0.02};
  Point3Distance* m = new (std::aligned_alloc(64, 512)) Point3Distance();
  m->setState(x);
  const size_t ce = chunkElements(n, 3, 6), rows = ce * 3;
  double* scratch = (double*)std::aligned_alloc(64, 8 * 8192);
  double* res = scratch; double* jac = res + rows; double* ej = jac + rows * 6;
  ChunkEvaluator<Point3Distance> ev{m, x, c.source.data(), c.target.data(), 3, 3, 6};
  const size_t chunks = (n + ce - 1) / ce;
  auto loop = [&](bool r, bool j, bool sq, bool acc) {
    double H[144] = {}, g[12] = {}, cost = 0;
    for (size_t k = 0; k < chunks; ++k) {
      size_t first = k * ce, count = std::min(ce, n - first);
      Index rr = count * 3;
      if (r) ev.evaluateResiduals(first, count, res);
      if (j) ev.fillJacobian(first, count, ej, jac, rows);
      if (sq) cost += squaredNorm(res, rr);
      if (acc) accumulateNormalEquations(jac, rows, res, rr, 6, H, 6, g);
    }
    sink = cost + H[0] + g[0];
  };
  std::printf("res           %8.3f ms\n", best([&] { loop(1,0,0,0); }) / 1e6);
  std::printf("jac           %8.3f ms\n", best([&] { loop(0,1,0,0); }) / 1e6);
  std::printf("sq            %8.3f ms\n", best([&] { loop(0,0,1,0); }) / 1e6);
  std::printf("acc           %8.3f ms\n", best([&] { loop(0,0,0,1); }) / 1e6);
  std::printf("res+jac       %8.3f ms\n", best([&] { loop(1,1,0,0); }) / 1e6);
  std::printf("all           %8.3f ms\n", best([&] { loop(1,1,1,1); }) / 1e6);
}
