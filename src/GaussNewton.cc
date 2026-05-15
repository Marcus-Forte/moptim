#include "moptim/GaussNewton.h"

#include <cmath>

#include <Eigen/Dense>

#include "Timer.hh"
#include "moptim/Convergence.h"
#include "moptim/EigenSolver.h"

namespace moptim {

template <class T, class PlusOperator>
	requires IPlus<PlusOperator, T>
GaussNewton<T, PlusOperator>::GaussNewton(
		size_t dimensions, const std::shared_ptr<ILog>& logger,
		const std::shared_ptr<ISolver<T>>& solver)
		: IOptimizer<T>(dimensions), solver_(solver), logger_(logger) {}

template <class T, class PlusOperator>
	requires IPlus<PlusOperator, T>
GaussNewton<T, PlusOperator>::GaussNewton(size_t dimensions,
																					const std::shared_ptr<ILog>& logger)
		: IOptimizer<T>(dimensions),
			solver_(std::make_shared<EigenSolver<T>>(logger, dimensions)),
			logger_(logger) {}

template <class T, class PlusOperator>
	requires IPlus<PlusOperator, T>
Status GaussNewton<T, PlusOperator>::step(T* x) const {
	using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	MatrixT JTJ(this->dimensions_, this->dimensions_);
	VectorT JTb(this->dimensions_);

	MatrixT Hessian = MatrixT::Zero(this->dimensions_, this->dimensions_);
	VectorT BVec = VectorT::Zero(this->dimensions_);
	VectorT DeltaVec(this->dimensions_);
	Eigen::Map<VectorT> XVec(x, this->dimensions_);
	T totalCost = 0.0;

	for (const auto& cost : this->costs_) {
		T cost_val = 0.0;
		cost->computeLinearSystem(x, JTJ.data(), JTb.data(), cost_val);
		Hessian += JTJ;
		BVec += JTb;
		totalCost += cost_val;
	}

	solver_->solve(Hessian.data(), BVec.data(), DeltaVec.data());
	VectorT XiVec(this->dimensions_);
	PlusOperator::plus(x, DeltaVec.data(), XiVec.data(), this->dimensions_);
	XVec = XiVec;

	logger_->log(ILog::Level::DEBUG, " Cost: {} ", totalCost);

	if (isCostSmall(totalCost)) {
		return Status::CONVERGED;
	}

	if (isDeltaSmall(DeltaVec.data(), DeltaVec.size())) {
		logger_->log(ILog::Level::DEBUG, " Delta < {} ",
								 std::sqrt(std::numeric_limits<T>::epsilon()));
		return Status::SMALL_DELTA;
	}

	return Status::STEP_OK;
}

template <class T, class PlusOperator>
	requires IPlus<PlusOperator, T>
Status GaussNewton<T, PlusOperator>::optimize(T* x) const {
	for (size_t i = 0; i < this->max_iterations_; ++i) {
		static Timer timer;
		const auto delta = timer.stop();
		logger_->log(ILog::Level::DEBUG, "GN Iteration: {}/{} (took: {} us)",
								 i + 1, this->max_iterations_, delta);
		timer.start();
		const auto status = step(x);

		if (status != Status::STEP_OK) {
			return status;
		}
	}
	return Status::MAX_ITERATIONS_REACHED;
}

template class GaussNewton<double, EuclideanPlusOperator<double>>;
template class GaussNewton<float, EuclideanPlusOperator<float>>;

}  // namespace moptim