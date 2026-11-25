#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "moptim/ExecutionContext.hh"
#include "moptim/ICost.hh"

namespace moptim {

/**
 * @brief Several costs of one problem evaluated together, in one dispatch.
 *
 * Every evaluation of a cost is dispatched through the ExecutionContext, which has a fixed price (a
 * function call inline, a task-arena hand-off when it runs on several threads) that small problems pay many
 * times over: a scan registration with a point-to-plane cost and an IMU factor dispatches twice per
 * evaluation. A FusedCost holds such costs and evaluates them in a single dispatch: each worker works
 * through the partitions of the first cost, then of the second, and so on. Each cost keeps its own
 * partitions, partial sums, scratch and model, so its linear system and cost are bit-identical to its own
 * evaluation's, and the FusedCost adds them in the
 * order the costs were given. Added to an optimizer in place of its parts, it gives the same `x`, the
 * same iterations and the same final cost, bit for bit, with one dispatch per evaluation instead of one
 * per cost.
 *
 * `Costs` are ChunkedCost types (AnalyticalCost, NumericalCost*), all built on the same
 * ExecutionContext and the same parameter space (the same manifold object, or plain
 * dimensions of the same size). The sub-costs stay usable on their own (the FusedCost shares them),
 * but must not be added to the optimizer as well, and must not be evaluated while the FusedCost has an
 * evaluation pending.
 */
template <class T, class... Costs>
class FusedCost final : public ICost<T> {
  static_assert(sizeof...(Costs) >= 1, "a FusedCost needs at least one cost");

 public:
  /// @throws std::invalid_argument if a cost is null, or the costs do not share their execution context
  ///   and parameter space.
  explicit FusedCost(std::shared_ptr<Costs>... costs)
      : ICost<T>(0, 0, spaceOfFirst(costs...), (checkNotNull(costs)->numElements() + ...),
                 std::max({checkNotNull(costs)->activeParamDim()...})),
        costs_(std::move(costs)...),
        context_(std::get<0>(costs_)->context()) {
    std::apply(
        [this](const auto&... cost) {
          const bool same = ((cost->context() == context_ && cost->manifold() == this->manifold() &&
                              cost->paramDim() == this->paramDim() && cost->tangentDim() == this->tangentDim()) &&
                             ...);
          if (!same) {
            throw std::invalid_argument(
                "moptim: the costs of a FusedCost must share their ExecutionContext and parameter space");
          }
        },
        costs_);
    for (Workspace& workspace : workspaces_) {
      workspace.hessian.resize(this->tangent_dim_, this->tangent_dim_);
      workspace.gradient.resize(this->tangent_dim_);
    }
  }

  ~FusedCost() override {
    try {
      wait();
    } catch (...) {  // NOLINT: nowhere to report it from a destructor
    }
  }

  T computeCost(const T* x) override {
    T cost{0};
    submitCost(x, &cost);
    wait();
    return cost;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) override {
    submitLinearSystem(x, JTJ, JTb, &cost);
    wait();
  }

  void submitCost(const T* x, T* cost) override { submit<false>(x, nullptr, nullptr, cost); }

  void submitLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) override { submit<true>(x, JTJ, JTb, cost); }

  /// Completes the outstanding evaluation: lets each cost add its partials, then sums the costs'
  /// results in their order.
  void wait() override {
    if (pending_.kind == Kind::None) {
      return;
    }
    const Pending pending = pending_;
    pending_.kind = Kind::None;
    std::apply([](auto&... cost) { (cost->finishFused(), ...); }, costs_);

    const size_t n = this->tangent_dim_;
    T total = workspaces_[0].cost;
    if (pending.kind == Kind::LinearSystem) {
      Eigen::Map<MatrixT> hessian(pending.JTJ, n, n);
      Eigen::Map<VectorT> gradient(pending.JTb, n);
      hessian = workspaces_[0].hessian;
      gradient = workspaces_[0].gradient;
      for (size_t i = 1; i < kCount; ++i) {
        hessian += workspaces_[i].hessian;
        gradient += workspaces_[i].gradient;
        total += workspaces_[i].cost;
      }
    } else {
      for (size_t i = 1; i < kCount; ++i) {
        total += workspaces_[i].cost;
      }
    }
    *pending.cost = total;
  }

  /// The `I`-th cost.
  template <size_t I>
  const auto& cost() const {
    return std::get<I>(costs_);
  }

  const std::shared_ptr<ExecutionContext>& context() const { return context_; }

 private:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  static constexpr size_t kCount = sizeof...(Costs);

  enum class Kind { None, Cost, LinearSystem };
  struct Pending {
    Kind kind = Kind::None;
    T* JTJ = nullptr;
    T* JTb = nullptr;
    T* cost = nullptr;
  };
  /// Where one cost writes its results.
  struct Workspace {
    MatrixT hessian;
    VectorT gradient;
    T cost{0};
  };

  template <class C>
  static const std::shared_ptr<C>& checkNotNull(const std::shared_ptr<C>& cost) {
    if (!cost) {
      throw std::invalid_argument("moptim: null cost given to a FusedCost");
    }
    return cost;
  }

  template <class First, class... Rest>
  static typename ICost<T>::ParamSpace spaceOfFirst(const std::shared_ptr<First>& first,
                                                    const std::shared_ptr<Rest>&... /*rest*/) {
    checkNotNull(first);
    return {first->paramDim(), first->tangentDim(), first->sharedManifold()};
  }

  template <bool kLinearSystem>
  void submit(const T* x, T* JTJ, T* JTb, T* cost) {
    wait();
    pending_ = Pending{kLinearSystem ? Kind::LinearSystem : Kind::Cost, JTJ, JTb, cost};
    try {
      submitFused<kLinearSystem>(x, std::index_sequence_for<Costs...>{});
    } catch (...) {
      settle();
      throw;
    }
  }

  template <bool kLinearSystem, size_t... Is>
  void submitFused(const T* x, std::index_sequence<Is...>) {
    const size_t width = std::max({std::get<Is>(costs_)->template chunkedWidth<kLinearSystem>()...});
    const std::array<bool, kCount> launched = {std::get<Is>(costs_)->template prepareFused<kLinearSystem>(
        x, width, workspaces_[Is].hessian.data(), workspaces_[Is].gradient.data(), &workspaces_[Is].cost)...};
    if (!(launched[Is] || ...)) {
      return;  // no elements anywhere: the outputs are written, nothing to wait for
    }
    const std::tuple<typename Costs::FusedRef...> refs{std::get<Is>(costs_)->fusedRef()...};
    detail::submit(*context_, width, [refs, launched](size_t worker) {
      ((launched[Is] ? Costs::template runFused<kLinearSystem>(std::get<Is>(refs), worker) : void()), ...);
    });
  }

  /// After a failed submit: completes whatever was prepared, so no cost is left with a pending evaluation.
  void settle() noexcept {
    pending_.kind = Kind::None;
    std::apply(
        [](auto&... cost) {
          ((
               [&] {
                 try {
                   cost->wait();
                 } catch (...) {  // NOLINT
                 }
               }()),
           ...);
        },
        costs_);
  }

  std::tuple<std::shared_ptr<Costs>...> costs_;
  std::shared_ptr<ExecutionContext> context_;
  std::array<Workspace, kCount> workspaces_;
  Pending pending_;
};

/// Builds a FusedCost from shared costs, deducing their types.
template <class T, class... Costs>
std::shared_ptr<FusedCost<T, Costs...>> makeFusedCost(std::shared_ptr<Costs>... costs) {
  return std::make_shared<FusedCost<T, Costs...>>(std::move(costs)...);
}

}  // namespace moptim
