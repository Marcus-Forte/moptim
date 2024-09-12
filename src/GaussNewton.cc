#include "GaussNewton.hh"

constexpr double g_small_cost = 1e-80;

GaussNewton::GaussNewton() = default;
GaussNewton::GaussNewton(const std::shared_ptr<ILog>& logger) : IOptimizer(logger) {}

IOptimizer::Status GaussNewton::step(Eigen::VectorXd& x) const {
  Eigen::MatrixXd Hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::VectorXd BVec = Eigen::VectorXd::Zero(x.size());
  Eigen::VectorXd delta(x.size());
  double totalCost = 0.0;

  for (const auto& cost : costs_) {
    const auto& [JtJ_, Jtb_, cost_val] = cost->computeLinearSystem(x);
    Hessian += JtJ_;
    BVec += Jtb_;
    totalCost += cost_val;
  }

  Eigen::LDLT<Eigen::MatrixXd> solver(Hessian);
  delta = solver.solve(-BVec);
  x += delta;

  if (logger_) {
    std::stringstream delta_str;
    delta_str << delta.transpose();
    logger_->log(ILog::Level::DEBUG, "delta: [{}], Cost: {} ", delta_str.str(), totalCost);
  }

  if (totalCost < g_small_cost) {
    return Status::CONVERGED;
  }

  if (isDeltaSmall(delta)) {
    return Status::SMALL_DELTA;
  }

  return Status::STEP_OK;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations, cost
IOptimizer::Status GaussNewton::optimize(Eigen::VectorXd& x) const {
  for (int i = 0; i < max_iterations_; i++) {
    if (logger_) {
      logger_->log(ILog::Level::DEBUG, "GN Iteration: {}/{}", i, max_iterations_);
    }
    const auto status = step(x);

    if (status != Status::STEP_OK) {
      return status;
    }
  }
  return IOptimizer::Status::MAX_ITERATIONS_REACHED;
}