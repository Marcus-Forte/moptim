#include "moptim/GaussNewton.hh"

#include "moptim/Convergence.hh"
#include "moptim/EigenSolver.hh"

namespace moptim {
template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), solver_(solver), delta_(dimensions) {}

template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions)
    : GaussNewton(dimensions, std::make_shared<EigenSolver<T>>(dimensions)) {}

template <class T>
typename GaussNewton<T>::StepResult GaussNewton<T>::stepImpl(T* x, size_t iteration) {
  using Clock = typename IOptimizer<T>::Clock;

  const auto start = this->now();
  const T cost = this->assembleLinearSystem(x, iteration);

  const auto emit = [&](Phase phase, Status status) {
    if (!this->observer_) {
      return;
    }
    IterationEvent<T> event;
    event.iteration = iteration;
    event.phase = phase;
    event.status = status;
    event.cost = cost;
    event.delta_norm = delta_.norm();
    event.elapsed = Clock::now() - start;
    this->observer_->onIteration(event);
  };

  if (!solver_->solve(this->hessian_.data(), this->gradient_.data(), delta_.data())) {
    // No usable step: x is left where it was, so `cost` is still the cost at x.
    delta_.setZero();
    emit(Phase::REJECTED, Status::NUMERIC_ERROR);
    return {Status::NUMERIC_ERROR, cost};
  }

  Eigen::Map<VectorT>(x, this->dimensions_) += delta_;

  Status status = Status::STEP_OK;
  if (isCostSmall(cost)) {
    status = Status::CONVERGED;
  } else if (isDeltaSmall(delta_.data(), delta_.size())) {
    status = Status::SMALL_DELTA;
  }

  emit(Phase::ACCEPTED, status);
  return {status, std::nullopt};
}

template class GaussNewton<double>;
template class GaussNewton<float>;

}  // namespace moptim
