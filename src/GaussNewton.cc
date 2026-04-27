#include "GaussNewton.hh"

#include <cassert>
#include <cmath>

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
Status GaussNewton<T>::step(std::span<T> x) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  assert(x.size() == this->dimensions_);

  MatrixT JTJ(this->dimensions_, this->dimensions_);
  VectorT JTb(this->dimensions_, 1);

  MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
  VectorT BVec = VectorT::Zero(this->dimensions_);
  VectorT DeltaVec(this->dimensions_);
  Eigen::Map<VectorT> XVec(x.data(), this->dimensions_);
  T totalCost = 0.0;

  // Compute Hessian
  for (const auto& cost : this->costs_) {
    T cost_val = 0.0;
    cost->computeLinearSystem(x, std::span<T>(JTJ.data(), JTJ.size()), std::span<T>(JTb.data(), JTb.size()), cost_val);
    Hessian += JTJ;
    BVec += JTb;
    totalCost += cost_val;
  }

  solver_->solve(std::span<const T>(Hessian.data(), Hessian.size()), std::span<const T>(BVec.data(), BVec.size()),
                 std::span<T>(DeltaVec.data(), DeltaVec.size()));
  XVec += DeltaVec;

  logger_->log(ILog::Level::DEBUG, " Cost: {} ", totalCost);

  if (isCostSmall(totalCost)) {
    return Status::CONVERGED;
  }

  if (isDeltaSmall(std::span<const T>(DeltaVec.data(), DeltaVec.size()))) {
    logger_->log(ILog::Level::DEBUG, " Delta < {} ", std::sqrt(std::numeric_limits<T>::epsilon()));
    return Status::SMALL_DELTA;
  }

  return Status::STEP_OK;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations, cost
template <class T>
Status GaussNewton<T>::optimize(std::span<T> x) const {
  for (size_t i = 0; i < this->max_iterations_; ++i) {
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
template class GaussNewton<float>;

}  // namespace moptim