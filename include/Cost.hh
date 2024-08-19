#pragma once

#include <functional>

#include "ICost.hh"

class Cost : public ICost {
 public:
  double sum(const double* x) const override;

 private:
};