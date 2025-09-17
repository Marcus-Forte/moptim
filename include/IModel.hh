#pragma once

template <class T>
class IModel {
 public:
  virtual ~IModel() = default;
  /**
   * @brief Setup / prepare the model with given parameters
   *
   * @param x
   */
  virtual void setup(const T* x) = 0;

  /**
   * @brief Compute the model output f(x) for given input and measurement
   *
   * @param input
   * @param measurement
   * @param[out] f_x
   */
  virtual void f(const T* input, const T* measurement, T* f_x) = 0;
};

template <class T>
class IJacobianModel : public IModel<T> {
 public:
  ~IJacobianModel() override = default;

  /**
   * @brief Compute the model Jacobian df/dx for given input and measurement
   *
   * @param input
   * @param measurement
   * @param[out] df_x
   */
  virtual void df(const T* input, const T* measurement, T* df_x) = 0;
};