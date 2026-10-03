#include "moptim/GaussNewton.hh"

#include <chrono>
#include <cmath>

#include "moptim/Convergence.hh"
#include "moptim/EigenSolver.hh"

namespace moptim {
template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions, const std::shared_ptr<ISolver<T>>& solver)
    : IOptimizer<T>(dimensions), solver_(solver) {}

template <class T>
GaussNewton<T>::GaussNewton(size_t dimensions)
    : IOptimizer<T>(dimensions), solver_(std::make_shared<EigenSolver<T>>(dimensions)) {}

template <class T>
Status GaussNewton<T>::step(T* x) const {
  return stepImpl(x, 0);
}

template <class T>
Status GaussNewton<T>::stepImpl(T* x, size_t iteration) const {
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  MatrixT JTJ(this->dimensions_, this->dimensions_);
  VectorT JTb(this->dimensions_);

  MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
  VectorT BVec = VectorT::Zero(this->dimensions_);
  VectorT DeltaVec(this->dimensions_);
  Eigen::Map<VectorT> XVec(x, this->dimensions_);
  T totalCost = 0.0;

  const auto start = std::chrono::steady_clock::now();

  // Compute Hessian
  for (const auto& cost : this->costs_) {
    T cost_val = 0.0;
    cost->computeLinearSystem(x, JTJ.data(), JTb.data(), cost_val);
    Hessian += JTJ;
    BVec += JTb;
    totalCost += cost_val;
  }

  if (this->observer_) {
    LinearSystemEvent<T> event;
    event.iteration = iteration;
    event.cost = totalCost;
    event.elapsed = std::chrono::steady_clock::now() - start;
    this->observer_->onLinearSystem(event);
  }

  solver_->solve(Hessian.data(), BVec.data(), DeltaVec.data());
  XVec += DeltaVec;

  Status status = Status::STEP_OK;
  if (isCostSmall(totalCost)) {
    status = Status::CONVERGED;
  } else if (isDeltaSmall(DeltaVec.data(), DeltaVec.size())) {
    status = Status::SMALL_DELTA;
  }

  if (this->observer_) {
    IterationEvent<T> event;
    event.iteration = iteration;
    event.phase = Phase::ACCEPTED;
    event.status = status;
    event.cost = totalCost;
    event.delta_norm = DeltaVec.norm();
    event.elapsed = std::chrono::steady_clock::now() - start;
    this->observer_->onIteration(event);
  }

  return status;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations, cost
template <class T>
Result<T> GaussNewton<T>::optimize(T* x) const {
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

  if (this->observer_) {
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

template class GaussNewton<double>;
template class GaussNewton<float>;

}  // namespace moptim
