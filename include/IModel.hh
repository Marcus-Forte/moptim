
#pragma once

#include <cstddef>

/**
 * @brief f(x, input) = measurement
 *
 */

class IModel {
 public:
  virtual void setup(const double* x) = 0;
  virtual void f(const double* input, const double* measurement, double* f_x) = 0;
};