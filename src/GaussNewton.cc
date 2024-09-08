#include "GaussNewton.hh"

constexpr double g_small_cost = 1e-80;

GaussNewton::GaussNewton() = default;
GaussNewton::GaussNewton(const std::shared_ptr<ILog>& logger) : IOptimizer(logger) {}

double GaussNewton::step(Eigen::VectorXd& x) const {
  Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::VectorXd Jtb = Eigen::VectorXd::Zero(x.size());

  double totalCost = 0.0;
  for (const auto& cost : costs_) {
    const auto& [JtJ_, Jtb_, cost_val] = cost->computeHessian(x);
    JtJ += JtJ_;
    Jtb += Jtb_;
    totalCost += cost_val;
  }

  Eigen::LDLT<Eigen::MatrixXd> solver(JtJ);
  const auto x_plus = solver.solve(-Jtb);
  x += x_plus;

  return totalCost;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations, cost
IOptimizer::Status GaussNewton::optimize(Eigen::VectorXd& x) const {
  for (int i = 0; i < max_iterations_; i++) {
    const auto total_error = step(x);

    if (logger_) {
      logger_->log(ILog::Level::INFO, std::format("Iteration: {}, Cost: {}", i, total_error));
    }
    if (isSmall(x)) {
      return IOptimizer::Status::SMALL_DELTA;
    }

    if (total_error < g_small_cost) {
      return IOptimizer::Status::CONVERGED;
    }
  }
  return IOptimizer::Status::MAX_ITERATIONS_REACHED;
}