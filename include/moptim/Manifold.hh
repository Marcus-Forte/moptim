#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace moptim {

/**
 * @brief The space the optimized parameters live in: a manifold embedded in `ambientDim()` numbers,
 * locally described by `tangentDim()` coordinates.
 *
 * The optimizers solve their linear system for a step `delta` of `tangentDim()` entries and move
 * the parameters with `plus`. The costs and models keep reading and writing the `ambientDim()`
 * numbers of `x`, so a pose can be stored in whatever form the consumer prefers (a matrix, a
 * quaternion and a translation, ...) and be optimized directly.
 *
 * This is a runtime interface on purpose, like ISolver and LossFunction: the optimizers are
 * explicitly instantiated in the library, so a template parameter would force every manifold of a
 * consumer to be instantiated there. The library ships only the Euclidean space and products of
 * spaces; rotations and poses, and the conventions that come with them, belong to the consumer.
 */
template <class T>
class IManifold {
 public:
  virtual ~IManifold() = default;

  /// Number of entries of `x`, the parameter vector costs and models read.
  virtual size_t ambientDim() const = 0;

  /// Number of entries of a step: the size of J^T J, J^T b and of the step `delta`.
  virtual size_t tangentDim() const = 0;

  /// `x_plus = x [+] delta`. `x` and `x_plus` have `ambientDim()` entries and `delta` has
  /// `tangentDim()`. `x_plus` does not alias `x` or `delta`. `plus(x, 0)` must return `x`.
  virtual void plus(const T* x, const T* delta, T* x_plus) const = 0;
};

/// Plain vector space: `x [+] delta = x + delta`, with `ambientDim() == tangentDim()`.
template <class T>
class EuclideanManifold final : public IManifold<T> {
 public:
  explicit EuclideanManifold(size_t dim) : dim_(dim) {}

  size_t ambientDim() const override { return dim_; }
  size_t tangentDim() const override { return dim_; }

  void plus(const T* x, const T* delta, T* x_plus) const override {
    for (size_t i = 0; i < dim_; ++i) {
      x_plus[i] = x[i] + delta[i];
    }
  }

 private:
  size_t dim_;
};

/**
 * @brief Cartesian product of manifolds, e.g. a pose and a velocity/bias vector.
 *
 * The blocks follow one another in `x` and, in the same order, in `delta`.
 */
template <class T>
class ProductManifold final : public IManifold<T> {
 public:
  explicit ProductManifold(std::vector<std::shared_ptr<const IManifold<T>>> blocks) : blocks_(std::move(blocks)) {
    ambient_offsets_.reserve(blocks_.size());
    tangent_offsets_.reserve(blocks_.size());
    for (const auto& block : blocks_) {
      assert(block);
      ambient_offsets_.push_back(ambient_dim_);
      tangent_offsets_.push_back(tangent_dim_);
      ambient_dim_ += block->ambientDim();
      tangent_dim_ += block->tangentDim();
    }
  }

  size_t ambientDim() const override { return ambient_dim_; }
  size_t tangentDim() const override { return tangent_dim_; }

  void plus(const T* x, const T* delta, T* x_plus) const override {
    for (size_t i = 0; i < blocks_.size(); ++i) {
      blocks_[i]->plus(x + ambient_offsets_[i], delta + tangent_offsets_[i], x_plus + ambient_offsets_[i]);
    }
  }

 private:
  std::vector<std::shared_ptr<const IManifold<T>>> blocks_;
  std::vector<size_t> ambient_offsets_;  ///< First entry of each block in x.
  std::vector<size_t> tangent_offsets_;  ///< First entry of each block in delta.
  size_t ambient_dim_ = 0;
  size_t tangent_dim_ = 0;
};

}  // namespace moptim
