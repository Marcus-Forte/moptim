#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

#include "ICost.hh"

template <class InputT, class OutputT, class Model>
class BaseCost : public ICost {
 public:
  using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  BaseCost(const BaseCost&) = delete;
  // No dataset: parameter only optimization. Initialize dummy iterators.
  BaseCost(size_t param_dim)
      : param_dim_(param_dim),
        input_{new std::vector<InputT>{{}}},
        observations_{new std::vector<OutputT>{{}}},
        no_input_(true) {
    jacobian_.resize(sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(sizeof(OutputT) / sizeof(double));
    hessian_.resize(param_dim_, param_dim_);
    b_.resize(param_dim_);
  }
  BaseCost(const std::vector<InputT>* input, const std::vector<OutputT>* observations, size_t param_dim)
      : input_(input), observations_(observations), param_dim_(param_dim), no_input_(false) {
    jacobian_.resize(input_->size() * sizeof(OutputT) / sizeof(double), param_dim_);
    residual_.resize(input_->size() * sizeof(OutputT) / sizeof(double));
    hessian_.resize(param_dim_, param_dim_);
    b_.resize(param_dim_);
  }

  ~BaseCost() {
    if (no_input_) {
      delete input_;
      delete observations_;
    }
  }

  Summation computeResidual(const Eigen::VectorXd& x) override {
    auto* residual_ptr = reinterpret_cast<OutputT*>(residual_.data());
    std::transform(input_->begin(), input_->end(), observations_->begin(), residual_ptr, Model(x));
    auto total_cost = residual_.squaredNorm();
    return {residual_, total_cost};
  }

  // TODO covariance
  SolveRhs computeHessian(const Eigen::VectorXd& x) override {
    const auto [residual, total_cost] = computeResidual(x);
    const auto jacobian = computeJacobian(x);
    hessian_ = jacobian.transpose() * jacobian;
    b_ = jacobian.transpose() * residual;
    return {hessian_, b_, total_cost};
  }

 protected:
  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;
  const size_t param_dim_;
  JacobianType jacobian_;
  Eigen::VectorXd residual_;
  Eigen::MatrixXd hessian_;
  Eigen::VectorXd b_;

  bool no_input_;
};