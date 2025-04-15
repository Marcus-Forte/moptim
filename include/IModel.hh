#pragma once

/**
 * @brief f(x, input) = measurement
 *
 */

class IModel {
 public:
  virtual ~IModel() = default;
  virtual void setup(const double* x) = 0;
  virtual void f(const double* input, const double* measurement, double* f_x) = 0;
};

class IJacobianModel : public IModel {
 public:
  ~IJacobianModel() override = default;
  virtual void df(const double* input, const double* measurement, double* df_x) = 0;
};