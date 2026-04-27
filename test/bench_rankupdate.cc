#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

// Benchmark: full DGEMM  vs  selfadjointView rankUpdate
// for J^T J where J is [num_residuals x param_dim].

template <int P>
void bench(const char* label, int num_residuals, int reps) {
  using MatrixT = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-1, 1);

  MatrixT J(num_residuals, P);
  for (int i = 0; i < J.size(); ++i) J(i) = dist(rng);

  MatrixT JTJ_full(P, P);
  MatrixT JTJ_rank(P, P);

  // warm up
  for (int i = 0; i < 100; ++i) {
    JTJ_full.noalias() = J.transpose() * J;
    JTJ_rank.setZero();
    JTJ_rank.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose());
  }

  // --- full DGEMM ---
  volatile double sink = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) {
    JTJ_full.noalias() = J.transpose() * J;
    sink += JTJ_full(0, 0);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double full_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;

  // --- rankUpdate (DSYRK) ---
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) {
    JTJ_rank.setZero();
    JTJ_rank.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose());
    sink += JTJ_rank(0, 0);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double rank_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / reps;

  (void)sink;
  std::cout << label << "  param_dim=" << P << "  residuals=" << num_residuals
            << "  full=" << full_ns << " ns  rankUpdate=" << rank_ns << " ns"
            << "  speedup=" << full_ns / rank_ns << "x\n";
}

int main() {
  const int reps = 100000;
  std::cout << "J^T J benchmark (avg over " << reps << " reps)\n";
  std::cout << "---------------------------------------------------\n";
  bench<2> ("tiny  ", 200, reps);
  bench<3> ("small ", 200, reps);
  bench<6> ("medium", 200, reps);
  bench<12>("large ", 200, reps);
}
