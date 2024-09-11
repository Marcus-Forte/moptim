#include "LevenbergMarquardt.hh"

constexpr double g_small_cost = 1e-80;

LevenbergMarquardt::LevenbergMarquardt() = default;

LevenbergMarquardt::LevenbergMarquardt(const std::shared_ptr<ILog>& logger) : IOptimizer(logger) {}

double LevenbergMarquardt::step(Eigen::VectorXd& x) const {
  Eigen::MatrixXd Hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::MatrixXd HessianDiagnonal(x.size(), x.size());
  Eigen::VectorXd BVec = Eigen::VectorXd::Zero(x.size());

  double totalCost = 0.0;
  for (const auto& cost : costs_) {
    const auto& [JtJ_, Jtb_, cost_val] = cost->computeLinearSystem(x);
    Hessian += JtJ_;
    BVec += Jtb_;
    totalCost += cost_val;
  }

  HessianDiagnonal = Hessian.diagonal().asDiagonal();
  for (int i = 0; i < lm_iterations_; ++i) {
    Eigen::LDLT<Eigen::MatrixXd> solver(Hessian + lm_lambda_ * HessianDiagnonal);
  }

  Eigen::LDLT<Eigen::MatrixXd> solver(Hessian);
  const auto x_plus = solver.solve(-BVec);
  x += x_plus;

  return totalCost;
}

IOptimizer::Status LevenbergMarquardt::optimize(Eigen::VectorXd& x) const {
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
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