#include "moptim/LevenbergMarquardt.hh"

#include <algorithm>
#include <cmath>

#include "moptim/Convergence.hh"
#include "moptim/EigenSolver.hh"

namespace moptim {

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), solver_(solver), diagonal_(dimensions), delta_(dimensions), trial_x_(dimensions) {}

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(std::shared_ptr<const IManifold<T>> manifold,
                                          const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(std::move(manifold)),
      solver_(solver),
      diagonal_(this->tangent_dim_),
      delta_(this->tangent_dim_),
      trial_x_(this->ambient_dim_) {}

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(std::shared_ptr<const IManifold<T>> manifold)
    : LevenbergMarquardt(manifold, std::make_shared<EigenSolver<T>>(manifold->tangentDim())) {}

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions)
    : LevenbergMarquardt(dimensions, std::make_shared<EigenSolver<T>>(dimensions)) {}

template <class T>
typename LevenbergMarquardt<T>::StepResult LevenbergMarquardt<T>::stepImpl(T* x, size_t iteration) {
  using Clock = typename IOptimizer<T>::Clock;

  Eigen::Map<VectorT> x_map(x, this->ambient_dim_);

  // Iteration 0 starts a run (optimize() or a manual step()): the costs, their data or x may have
  // changed since the last call, so the system is always assembled afresh.
  if (iteration == 0 || !linear_system_is_current_) {
    linear_system_cost_ = this->assembleLinearSystem(x, iteration);
    diagonal_ = this->hessian_.diagonal();
    linear_system_is_current_ = true;
  }
  const T initial_cost = linear_system_cost_;

  if (!lambda_) {
    lambda_ = kInitialLambdaFactor * diagonal_.cwiseAbs().maxCoeff();
  }

  T nu = 2;
  for (size_t trial = 0; trial < kMaxTrials; ++trial) {
    const auto trial_start = this->now();
    const T lambda = *lambda_;

    const auto emit = [&](Phase phase, Status status, T cost, T rho) {
      if (!this->observer_) {
        return;
      }
      IterationEvent<T> event;
      event.iteration = iteration;
      event.trial = trial;
      event.phase = phase;
      event.status = status;
      event.cost = cost;
      event.previous_cost = initial_cost;
      event.rho = rho;
      event.lambda = lambda;
      event.delta_norm = delta_.norm();
      event.elapsed = Clock::now() - trial_start;
      this->observer_->onIteration(event);
    };
    const auto increaseDamping = [&] {
      *lambda_ *= nu;
      nu = 2 * nu;
    };

    // (J^T J + lambda * diag(J^T J)) * delta = -J^T b. Only the diagonal differs between trials.
    this->hessian_.diagonal() = diagonal_ + lambda * diagonal_;

    if (!solver_->solve(this->hessian_.data(), this->gradient_.data(), delta_.data())) {
      // The damped system is not positive definite yet: more damping may fix it.
      delta_.setZero();
      emit(Phase::REJECTED, Status::NUMERIC_ERROR, initial_cost, T{0});
      increaseDamping();
      continue;
    }

    this->applyStep(x, delta_.data(), trial_x_.data());
    const T trial_cost = this->evaluateCost(trial_x_.data());
    const T rho = (initial_cost - trial_cost) / delta_.dot(lambda * delta_ - this->gradient_);

    if (rho < 0 || std::isnan(rho)) {
      if (isDeltaSmall(delta_.data(), delta_.size())) {
        // x stays where it is, and so does its cost.
        const Status status = isCostSmall(trial_cost) ? Status::CONVERGED : Status::SMALL_DELTA;
        emit(Phase::FINISHED, status, trial_cost, rho);
        return {status, initial_cost};
      }

      emit(Phase::REJECTED, Status::STEP_OK, trial_cost, rho);
      increaseDamping();
      continue;
    }

    x_map = trial_x_;
    linear_system_is_current_ = false;
    *lambda_ *= static_cast<T>(std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3)));
    emit(Phase::ACCEPTED, Status::STEP_OK, trial_cost, rho);
    return {Status::STEP_OK, trial_cost};
  }

  // Every trial was rejected. x is untouched, so the linear system stays valid and the next
  // iteration of this run goes straight to trying larger damping factors.
  return {Status::STEP_OK, initial_cost};
}

template <class T>
void LevenbergMarquardt<T>::onOptimizeStart() {
  lambda_.reset();
  linear_system_is_current_ = false;
}

template <class T>
bool LevenbergMarquardt<T>::shouldEmitFinishedEvent(Status status) const {
  // stepImpl() emits FINISHED when it self-terminates (CONVERGED/SMALL_DELTA). Only emit here
  // when the iteration budget was exhausted, so the observer sees exactly one FINISHED event
  // per optimize() call.
  return status == Status::MAX_ITERATIONS_REACHED;
}

template class LevenbergMarquardt<double>;
template class LevenbergMarquardt<float>;
}  // namespace moptim
