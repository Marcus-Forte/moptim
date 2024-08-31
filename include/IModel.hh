#pragma once

class IModel {
 public:
  IModel(const double* x0) : x_(x0){};

 protected:
  const double* x_;
};