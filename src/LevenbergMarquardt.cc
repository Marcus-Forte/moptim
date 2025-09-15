#include "LevenbergMarquardt.hh"

#include "Convergence.hh"
#include "EigenSolver.hh"
#include "Timer.hh"

namespace moptim {

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ILog>& logger,
                                          const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), logger_(logger), solver_(solver) {}

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ILog>& logger)
    : IOptimizer<T>(dimensions), logger_(logger), solver_(std::make_shared<EigenSolver<T>>(logger, dimensions)) {}
template <class T>
Status LevenbergMarquardt<T>::step(T* x) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT JTJ(this->dimensions_, this->dimensions_);
  VectorT JTb(this->dimensions_, 1);

  MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
  MatrixT HessianDiagnonal = MatrixT::Zero(this->dimensions_, this->dimensions_);
  VectorT BVec = VectorT::Zero(this->dimensions_);
  VectorT XiVec = VectorT::Zero(this->dimensions_);
  VectorT DeltaVec(this->dimensions_);
  Eigen::Map<VectorT> XVec(x, this->dimensions_);

  T totalCost = 0.0;
  T initCost = 0.0;

  // Compute Hessian
  for (const auto& cost : this->costs_) {
    T cost_val = 0.0;
    cost->computeLinearSystem(x, JTJ.data(), JTb.data(), &cost_val);
    Hessian += JTJ;
    BVec += JTb;
    initCost += cost_val;
  }

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hessian.diagonal().array().abs().maxCoeff();
  }

  T nu = 2.0;

  HessianDiagnonal = Hessian.diagonal().asDiagonal();
  for (int i = 0; i < lm_iterations_; ++i) {
    // Refine Hessian
    Hessian += lm_lambda_ * HessianDiagnonal;

    solver_->solve(Hessian.data(), BVec.data(), DeltaVec.data());

    XiVec = XVec + DeltaVec;

    for (const auto& cost : this->costs_) {
      totalCost += cost->computeCost(XiVec.data());
    }
    auto rho = (initCost - totalCost) / DeltaVec.dot(lm_lambda_ * DeltaVec - BVec);

    // if (logger_) {
    //   std::stringstream delta_str;
    //   delta_str << delta.transpose();
    logger_->log(ILog::Level::DEBUG, "rho: {}, Cost: {} -> {}", rho, initCost, totalCost);
    // }

    if (rho < 0 || std::isnan(rho)) {
      if (isDeltaSmall(DeltaVec.data(), this->dimensions_)) {
        if (isCostSmall(totalCost)) {
          return Status::CONVERGED;
        }
        return Status::SMALL_DELTA;
      }

      lm_lambda_ *= nu;
      nu = 2 * nu;
      continue;
    }

    XVec = XiVec;
    lm_lambda_ *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    break;
  }

  return Status::STEP_OK;
}

template <class T>
Status LevenbergMarquardt<T>::optimize(T* x) const {
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
  static Timer timer;
  for (int i = 0; i < this->max_iterations_; i++) {
    timer.start();
    const auto status = step(x);

    if (logger_) {
      const auto delta = timer.stop();
      logger_->log(ILog::Level::DEBUG, "LM Iteration: {}/{} (took: {} us). Status: {}", i + 1, this->max_iterations_,
                   delta, static_cast<int>(status));
    }

    if (status != Status::STEP_OK) {
      return status;
    }
  }
  return Status::MAX_ITERATIONS_REACHED;
}

template class LevenbergMarquardt<double>;
template class LevenbergMarquardt<float>;
}  // namespace moptim