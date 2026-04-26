#pragma once

#include <span>

template <class T>
class IModel {
 public:
  virtual ~IModel() = default;
  /**
   * @brief Setup / prepare the model with given parameters
   *
   * @param x
   */
  virtual void setup(std::span<const T> x) = 0;

  /**
   * @brief Compute the model output f(x) for given input and measurement
   *
   * @param input
   * @param measurement
   * @param[out] f_x
   */
  virtual void f(std::span<const T> input, std::span<const T> measurement, std::span<T> f_x) = 0;
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
  virtual void df(std::span<const T> input, std::span<const T> measurement, std::span<T> df_x) = 0;
};