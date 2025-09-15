#pragma once

/**
 * @brief f(x, input) = measurement
 *
 */

template <class T>
class IModel {
 public:
  virtual ~IModel() = default;
  virtual void setup(const T* x) = 0;
  virtual void f(const T* input, const T* measurement, T* f_x) = 0;
};

template <class T>
class IJacobianModel : public IModel<T> {
 public:
  ~IJacobianModel() override = default;
  virtual void df(const T* input, const T* measurement, T* df_x) = 0;
};