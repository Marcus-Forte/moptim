#pragma once

class ICost {
 public:
  virtual double computeError(const double* x) const = 0;

 protected:
};