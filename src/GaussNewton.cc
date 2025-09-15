#include "GaussNewton.hh"

#include "Convergence.hh"
#include "EigenSolver.hh"
#include "Timer.hh"

namespace moptim {
template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions, const std::shared_ptr<ILog>& logger,
                            const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), logger_(logger), solver_(solver) {}

template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions, const std::shared_ptr<ILog>& logger)
    : IOptimizer<T>(dimensions), logger_(logger), solver_(std::make_shared<EigenSolver<T>>(logger, dimensions)) {}

template <class T>
Status GaussNewton<T>::step(T* x) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
  VectorT BVec = VectorT::Zero(this->dimensions_);
  VectorT DeltaVec(this->dimensions_);
  Eigen::Map<VectorT> XVec(x, this->dimensions_);
  T totalCost = 0.0;

  for (const auto& cost : this->costs_) {
    const auto& [JtJ_, Jtb_, cost_val] = cost->computeLinearSystem(XVec);
    Hessian += JtJ_;
    BVec += Jtb_;
    totalCost += cost_val;
  }

  solver_->solve(Hessian.data(), BVec.data(), DeltaVec.data());
  XVec += DeltaVec;

  logger_->log(ILog::Level::DEBUG, " Cost: {} ", totalCost);

  if (totalCost < moptim::constants::g_small_cost) {
    return Status::CONVERGED;
  }

  if (isDeltaSmall(DeltaVec.data(), this->dimensions_)) {
    return Status::SMALL_DELTA;
  }

  return Status::STEP_OK;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations, cost
template <class T>
Status GaussNewton<T>::optimize(T* x) const {
  for (int i = 0; i < this->max_iterations_; i++) {
    static Timer timer;
    const auto delta = timer.stop();
    logger_->log(ILog::Level::DEBUG, "GN Iteration: {}/{} (took: {} us)", i + 1, this->max_iterations_, delta);
    timer.start();
    const auto status = step(x);

    if (status != Status::STEP_OK) {
      return status;
    }
  }
  return Status::MAX_ITERATIONS_REACHED;
}

template class GaussNewton<double>;
// template class GaussNewton<float>;

}  // namespace moptim