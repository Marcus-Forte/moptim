#include "LevenbergMarquardt.hh"

#include <chrono>
#include <cmath>

#include "Convergence.hh"
#include "EigenSolver.hh"

namespace moptim {

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), solver_(solver) {}

template <class T>
LevenbergMarquardt<T>::LevenbergMarquardt(size_t dimensions)
    : IOptimizer<T>(dimensions), solver_(std::make_shared<EigenSolver<T>>(dimensions)) {}

template <class T>
Status LevenbergMarquardt<T>::step(T* x) const {
  return stepImpl(x, 0);
}

template <class T>
Status LevenbergMarquardt<T>::stepImpl(T* x, size_t iteration) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT JTJ(this->dimensions_, this->dimensions_);
  VectorT JTb(this->dimensions_);

  MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
  MatrixT HessianDiagnonal = MatrixT::Zero(this->dimensions_, this->dimensions_);
  VectorT BVec = VectorT::Zero(this->dimensions_);
  VectorT XiVec = VectorT::Zero(this->dimensions_);
  VectorT DeltaVec(this->dimensions_);
  Eigen::Map<VectorT> XVec(x, this->dimensions_);

  T initCost = 0.0;

  const auto start = std::chrono::steady_clock::now();

  // Compute Hessian
  for (const auto& cost : this->costs_) {
    T cost_val = 0.0;
    cost->computeLinearSystem(x, JTJ.data(), JTb.data(), cost_val);
    Hessian += JTJ;
    BVec += JTb;
    initCost += cost_val;
  }

  if (this->observer_) {
    LinearSystemEvent<T> event;
    event.iteration = iteration;
    event.cost = initCost;
    event.elapsed = std::chrono::steady_clock::now() - start;
    this->observer_->onLinearSystem(event);
  }

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hessian.diagonal().array().abs().maxCoeff();
  }

  T nu = 2.0;

  const MatrixT Hessian0 = Hessian;
  HessianDiagnonal = Hessian.diagonal().asDiagonal();
  T totalCost = 0.0;
  for (size_t i = 0; i < lm_iterations_; ++i) {
    const auto trial_start = std::chrono::steady_clock::now();
    const T trial_lambda = lm_lambda_;

    Hessian = Hessian0 + lm_lambda_ * HessianDiagnonal;

    solver_->solve(Hessian.data(), BVec.data(), DeltaVec.data());

    XiVec = XVec + DeltaVec;

    totalCost = 0.0;
    for (const auto& cost : this->costs_) {
      totalCost += cost->computeCost(XiVec.data());
    }
    auto rho = (initCost - totalCost) / DeltaVec.dot(lm_lambda_ * DeltaVec - BVec);

    const auto emit = [&](Phase phase, Status status) {
      if (!this->observer_) {
        return;
      }
      IterationEvent<T> event;
      event.iteration = iteration;
      event.trial = i;
      event.phase = phase;
      event.status = status;
      event.cost = totalCost;
      event.previous_cost = initCost;
      event.rho = rho;
      event.lambda = trial_lambda;
      event.delta_norm = DeltaVec.norm();
      event.elapsed = std::chrono::steady_clock::now() - trial_start;
      this->observer_->onIteration(event);
    };

    if (rho < 0 || std::isnan(rho)) {
      if (isDeltaSmall(DeltaVec.data(), DeltaVec.size())) {
        if (isCostSmall(totalCost)) {
          emit(Phase::FINISHED, Status::CONVERGED);
          return Status::CONVERGED;
        }
        emit(Phase::FINISHED, Status::SMALL_DELTA);
        return Status::SMALL_DELTA;
      }

      emit(Phase::REJECTED, Status::STEP_OK);
      lm_lambda_ *= nu;
      nu = 2 * nu;
      continue;
    }

    XVec = XiVec;
    lm_lambda_ *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    emit(Phase::ACCEPTED, Status::STEP_OK);
    break;
  }

  return Status::STEP_OK;
}

template <class T>
Result<T> LevenbergMarquardt<T>::optimize(T* x) const {
  lm_init_lambda_factor_ = 1e-7;
  lm_lambda_ = -1.0;

  Result<T> result;
  const auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < this->max_iterations_; ++i) {
    const auto status = stepImpl(x, i);
    result.iterations = i + 1;
    result.status = status;

    if (status != Status::STEP_OK) {
      break;
    }
  }

  if (result.status == Status::STEP_OK) {
    result.status = Status::MAX_ITERATIONS_REACHED;
  }

  result.final_cost = T{};
  for (const auto& cost : this->costs_) {
    result.final_cost += cost->computeCost(x);
  }

  // stepImpl emits FINISHED when it self-terminates (CONVERGED/SMALL_DELTA).
  // Only emit here when the iteration budget was exhausted, so the observer
  // sees exactly one FINISHED event per optimize() call.
  if (this->observer_ && result.status == Status::MAX_ITERATIONS_REACHED) {
    IterationEvent<T> event;
    event.iteration = result.iterations;
    event.phase = Phase::FINISHED;
    event.status = result.status;
    event.cost = result.final_cost;
    event.elapsed = std::chrono::steady_clock::now() - start;
    this->observer_->onIteration(event);
  }

  return result;
}

template class LevenbergMarquardt<double>;
template class LevenbergMarquardt<float>;
}  // namespace moptim
