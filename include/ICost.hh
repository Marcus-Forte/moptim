#pragma once

#include <cstddef>
class ICost {
 public:
  virtual double sum(const double* x) const = 0;

 protected:
  size_t param_dim_;
  size_t output_dim_;
  size_t input_dim_;
};