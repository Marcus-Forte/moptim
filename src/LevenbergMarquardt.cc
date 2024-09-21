#include "LevenbergMarquardt.hh"

#include "Timer.hh"

LevenbergMarquardt::LevenbergMarquardt() = default;
LevenbergMarquardt::LevenbergMarquardt(const std::shared_ptr<ILog>& logger) : IOptimizer(logger) {}

IOptimizer::Status LevenbergMarquardt::step(Eigen::VectorXd& x) const {
  Eigen::MatrixXd Hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::MatrixXd HessianDiagnonal(x.size(), x.size());
  Eigen::VectorXd BVec = Eigen::VectorXd::Zero(x.size());
  Eigen::VectorXd xi(x.size());
  Eigen::VectorXd delta(x.size());
  double totalCost = 0.0;
  double initCost = 0.0;

  // Compute Hessian
  for (const auto& cost : costs_) {
    const auto& [JtJ_, Jtb_, cost_val] = cost->computeLinearSystem(x);
    Hessian += JtJ_;
    BVec += Jtb_;
    initCost += cost_val;
  }

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hessian.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;

  HessianDiagnonal = Hessian.diagonal().asDiagonal();
  for (int i = 0; i < lm_iterations_; ++i) {
    // Refine Hessian
    Eigen::LDLT<Eigen::MatrixXd> solver(Hessian + lm_lambda_ * HessianDiagnonal);

    delta = solver.solve(-BVec);

    xi = x + delta;

    for (const auto& cost : costs_) {
      totalCost += cost->computeCost(xi);
    }
    auto rho = (initCost - totalCost) / delta.dot(lm_lambda_ * delta - BVec);

    if (logger_) {
      std::stringstream delta_str;
      delta_str << delta.transpose();
      logger_->log(ILog::Level::DEBUG, "rho: {}, delta: [{}], Cost: {} -> {}", rho, delta_str.str(), initCost,
                   totalCost);
    }

    if (rho < 0 || std::isnan(rho)) {
      if (isDeltaSmall(delta)) {
        if (totalCost < moptim::constants::g_small_cost) {
          return Status::CONVERGED;
        }
        return Status::SMALL_DELTA;
      }

      lm_lambda_ *= nu;
      nu = 2 * nu;
      continue;
    }

    x = xi;
    lm_lambda_ *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    break;
  }

  return Status::STEP_OK;
}

IOptimizer::Status LevenbergMarquardt::optimize(Eigen::VectorXd& x) const {
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
  for (int i = 0; i < max_iterations_; i++) {
    if (logger_) {
      static Timer timer;
      const auto delta = timer.stop();
      logger_->log(ILog::Level::DEBUG, "LM Iteration: {}/{} (took: {} us)", i + 1, max_iterations_, delta);
      timer.start();
    }

    const auto status = step(x);

    if (status != Status::STEP_OK) {
      return status;
    }
  }
  return Status::MAX_ITERATIONS_REACHED;
}