#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#include "ICost.hh"

template <class InputT, class OutputT, class Model>
class Cost : public ICost {
 public:
  Cost(InputT* input, OutputT* measurements, size_t num_elements, size_t param_dim)
      : input_(input), measurements_(measurements), num_elements_(num_elements), param_dim_(param_dim) {
    error_vec_ = new OutputT[num_elements_];
    error_vec_diff_ = new OutputT[num_elements_];
  }
  ~Cost() {
    delete[] error_vec_;
    delete[] error_vec_diff_;
  }
  double computeError(const double* x) const override {
    // Store error vector.
    std::transform(input_, input_ + num_elements_, measurements_, error_vec_, Model(x));
    return std::reduce(error_vec_, error_vec_ + num_elements_, 0.0, std::plus<>());
  }

  void computeJacobian(const double* x) {
    const Eigen::Map<const Eigen::VectorXd> x0(x, param_dim_);

    for (size_t i = 0; i < param_dim_; ++i) {
      Eigen::VectorXd x0_(x0);
      x0_[i] += 0.001;
      std::transform(input_, input_ + num_elements_, measurements_, error_vec_diff_, Model(x0_.data()));
    }
  }

  // double* getJacobian() {}

 private:
  const InputT* input_;
  const OutputT* measurements_;
  const size_t num_elements_;
  const size_t param_dim_;

  OutputT* error_vec_;
  OutputT* error_vec_diff_;
  Eigen::MatrixXd jacobian_;
};