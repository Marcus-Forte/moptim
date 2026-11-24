#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "moptim/ExecutionContext.hh"
#include "moptim/ICost.hh"
#include "moptim/IModel.hh"
#include "moptim/LossFunction.hh"
#include "moptim/detail/AlignedArray.hh"
#include "moptim/detail/ChunkKernel.hh"
#include "moptim/detail/Partitioning.hh"

namespace moptim {

template <class T, class... Costs>
class FusedCost;

/**
 * @brief Cost that streams its elements through cache-sized chunks, in one launch.
 *
 * The residuals and jacobian rows of a chunk of elements are evaluated into small buffers that stay
 * in the L1 data cache and are folded into J^T J, J^T b and the cost right away. The full jacobian
 * (num_elements * observation_dim rows) is never stored, so memory use and construction time do not
 * grow with the number of elements.
 *
 * Every evaluation (computeCost() and computeLinearSystem()) is one launch of the cost's
 * ExecutionContext, even for a handful of elements; with no elements there is no launch. The
 * chunks are grouped into partitions whose size depends only on the problem (detail/Partitioning.hh);
 * the work-items of the launch (run on oneTBB when the context allows several threads and the cost has
 * enough partitions, else inline on the calling thread) claim partitions, each writes one partial sum
 * per partition, and the partials are added in partition order. The result is therefore bit-identical
 * whatever number of threads the context allows, and a cost that fits in one partition reproduces a
 * sequential walk over its chunks exactly.
 *
 * `Policy` says how the jacobian rows of a chunk are obtained (AnalyticalJacobian in AnalyticalCost.hh,
 * NumericalJacobian in NumericalCost.hh). A policy with `kPointsPerParam` > 0 differentiates by finite
 * differences: the cost then keeps `1 + kPointsPerParam * active_param_dim` models, one per evaluation
 * point (point 0 is `x`), all copied from the one it was given. This class owns everything else: the optional noise model (scalar weights or
 * information matrices), the optional robust kernel, the model and the buffers of the launch.
 * It allocates them on first use and reuses them afterwards. A numerical cost
 * whose elements fit in one chunk and whose model is const-callable makes no copies: each evaluation
 * point is visited once, so the one model is moved from point to point (its state set on the way)
 * and the results are those of the copies, bit for bit.
 *
 * Only the built-in losses (LossFunction::descriptor() other than Custom) are evaluated.
 * submitCost() and submitLinearSystem() start the launch without waiting; wait() completes it and
 * writes the results. Not thread-safe: one evaluation at a time per cost.
 */
template <class Model, class T, class Policy>
  requires ResidualModel<Model, T>
class ChunkedCost : public ICost<T> {
 public:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  ~ChunkedCost() override {
    // A submitted evaluation is completed before the buffers it reads go.
    try {
      wait();
    } catch (...) {  // NOLINT: nowhere to report it from a destructor
    }
    destroyModels();
  }

  T computeCost(const T* x) final {
    T cost{0};
    submitCost(x, &cost);
    wait();
    return cost;
  }

  void computeLinearSystem(const T* x, T* JTJ, T* JTb, T& cost) final {
    submitLinearSystem(x, JTJ, JTb, &cost);
    wait();
  }

  /// Starts the launch of a cost-only evaluation and returns; wait() writes `cost`. A pending
  /// evaluation is completed first.
  void submitCost(const T* x, T* cost) final {
    wait();
    if (num_elements_ == 0) {
      *cost = T{0};
      return;
    }
    launch<false>(x);
    pending_ = Pending{Pending::Kind::Cost, nullptr, nullptr, cost, reduce_from_, reduce_count_};
  }

  void submitLinearSystem(const T* x, T* JTJ, T* JTb, T* cost) final {
    wait();
    if (num_elements_ == 0) {
      std::fill_n(JTJ, tangent_dim_ * tangent_dim_, T{0});
      std::fill_n(JTb, tangent_dim_, T{0});
      *cost = T{0};
      return;
    }
    launch<true>(x);
    pending_ = Pending{Pending::Kind::LinearSystem, JTJ, JTb, cost, reduce_from_, reduce_count_};
  }

  /// Completes the outstanding evaluation: waits for the launch and adds the partial results, in
  /// partition order, into the outputs given to submit*().
  void wait() final {
    if (pending_.kind == Pending::Kind::None) {
      return;
    }
    const Pending pending = pending_;
    pending_.kind = Pending::Kind::None;
    reduce(pending);
  }

  /**
   * @brief Attach a robust kernel (M-estimator) applied to each element's
   * (optionally whitened) squared residual norm. Passing nullptr (default)
   * disables it, falling back to plain least-squares. Shared ownership
   * allows the same kernel to be reused/adapted (e.g. online scale updates,
   * a la KISS-ICP) across multiple costs: its descriptor() is read at every
   * evaluation, so setScale() takes effect on the next one.
   *
   * @throws std::invalid_argument if the loss is a custom one (descriptor() of kind Custom): the
   *   kernel evaluates the built-in losses only.
   */
  void setLossFunction(std::shared_ptr<LossFunction<T>> loss) {
    wait();
    if (loss && loss->descriptor().kind == LossKind::Custom) {
      throw std::invalid_argument(
          "moptim: this cost evaluates its loss in a kernel, which supports the built-in losses only "
          "(LossFunction::descriptor() of a custom loss is LossKind::Custom)");
    }
    loss_ = std::move(loss);
  }

  /**
   * @brief Set a scalar, isotropic weight per element (w_i = 1/sigma_i^2).
   * Equivalent to setInformation() with information_i = w_i * I, but cheaper.
   * Clears any previously set information matrices.
   * `weights.size()` must equal the number of elements; an empty vector removes the weights.
   */
  void setWeights(const std::vector<T>& weights) {
    assert(weights.empty() || weights.size() == num_elements_);
    wait();
    information_factors_.reset();
    weights_.reset();
    if (!weights.empty()) {
      weights_ = detail::makeAlignedArray<T>(weights.size());
      std::copy(weights.begin(), weights.end(), weights_.get());
    }
  }

  /**
   * @brief Set a full per-element information matrix (inverse covariance),
   * e.g. propagated from a Kalman filter's measurement/process noise. Each
   * matrix is `observation_dim x observation_dim` and is factored once
   * (Cholesky) at set-time; residuals/jacobians are whitened with L^T.
   * Clears any previously set scalar weights.
   * `information.size()` must equal the number of elements; an empty vector removes the matrices.
   */
  void setInformation(const std::vector<MatrixT>& information) {
    assert(information.empty() || information.size() == num_elements_);
    wait();
    weights_.reset();
    information_factors_.reset();
    if (information.empty()) {
      return;
    }
    const size_t entries = observation_dim_ * observation_dim_;
    information_factors_ = detail::makeAlignedArray<T>(entries * information.size());
    for (size_t e = 0; e < information.size(); ++e) {
      assert(static_cast<size_t>(information[e].rows()) == observation_dim_ &&
             static_cast<size_t>(information[e].cols()) == observation_dim_);
      Eigen::Map<MatrixT>(information_factors_.get() + e * entries, observation_dim_, observation_dim_) =
          information[e].llt().matrixL();
    }
  }

  /**
   * @brief Replaces the elements: `num_elements` of them, read from `input` and `observations`
   * (same dimensions as before), for the next evaluations.
   *
   * The cost keeps its model, its loss and its buffers: they are reused, and only grown when the new
   * data needs more partitions or scratch than any earlier evaluation did. An iterative caller that
   * re-associates its data every round (ICP correspondences) updates one cost this way instead of
   * building a new one each round. Results are the ones a cost built with this data gives, bit for bit.
   *
   * The per-element noise model (setWeights(), setInformation()) belongs to the old elements and is
   * removed; set it again for the new ones. The arrays are read as they are at each evaluation, so they
   * must stay valid while the cost uses them. A pending evaluation is completed first.
   */
  void setData(const T* input, const T* observations, size_t num_elements) {
    wait();
    weights_.reset();
    information_factors_.reset();
    num_elements_ = num_elements;
    partitioning_ = detail::makePartitioning(num_elements_, observation_dim_, active_param_dim_, sizeof(T));
    const detail::SliceLayout layout = makeSlice(partitioning_);
    if (layout.stride > slice_layout_.stride) {
      scratch_.reset();
      scratch_slices_ = 0;  // ensureState() reallocates the slices at the new stride
    }
    slice_layout_ = layout;
    if (partials_ && partitioning_.num_partitions > partials_capacity_) {
      partials_ = detail::makeAlignedArray<T>(partitioning_.num_partitions * partial_stride_);
      partials_capacity_ = partitioning_.num_partitions;
    }
    input_ = input;
    observations_ = observations;
  }

  /**
   * @brief Replaces the model for the next evaluations, as if the cost had been built with `model`.
   *
   * The models are rebuilt in place from `model` (all the evaluation-point copies of
   * a numerical cost too), without reallocating. A pending evaluation is completed first.
   */
  void setModel(Model model) {
    wait();
    if (models_ == nullptr) {
      pending_model_.emplace(std::move(model));
      return;
    }
    Model* storage = models_;
    destroyModels();
    models_built_ = 0;
    pending_model_.emplace(std::move(model));
    constructModels(storage);
  }

  const std::shared_ptr<ExecutionContext>& context() const { return context_; }

 protected:
  ChunkedCost(const T* input, const T* observations, size_t num_elements, size_t input_dim, size_t observation_dim,
              typename ICost<T>::ParamSpace space, Model model, size_t active_param_dim,
              std::shared_ptr<ExecutionContext> context)
      : ICost<T>(input_dim, observation_dim, std::move(space), num_elements, active_param_dim),
        context_(context ? std::move(context) : defaultContext()),
        pending_model_(std::move(model)),
        model_count_(1 + Policy::kPointsPerParam * active_param_dim_),
        input_(input),
        observations_(observations),
        partitioning_(detail::makePartitioning(num_elements_, observation_dim_, active_param_dim_, sizeof(T))),
        partial_stride_(detail::partialStride(sizeof(T), active_param_dim_)),
        slice_layout_(makeSlice(partitioning_)) {}

  using ICost<T>::input_dim_;
  using ICost<T>::observation_dim_;
  using ICost<T>::param_dim_;
  using ICost<T>::tangent_dim_;
  using ICost<T>::num_elements_;
  using ICost<T>::active_param_dim_;

 private:
  static constexpr detail::ModelAccess kAccess = detail::modelAccess<Policy, Model>();

  /// Layout of a work-item's scratch slice for chunks of `partitioning`.
  detail::SliceLayout makeSlice(const detail::Partitioning& partitioning) const {
    return detail::makeSliceLayout(
        sizeof(T), partitioning.chunk_elements * observation_dim_, active_param_dim_,
        Policy::auxValues(partitioning.chunk_elements * observation_dim_, active_param_dim_, observation_dim_),
        kAccess == detail::ModelAccess::PerWorkItemCopy ? model_count_ * sizeof(Model) : 0, alignof(Model));
  }

  /// Allocates what a launch of `width` work-items needs and was not allocated yet: everything on the
  /// first call, more scratch slices if the width grew.
  void ensureState(size_t width) {
    if (models_ == nullptr) {
      buildModels();

      x_ = detail::makeAlignedArray<T>(param_dim_);
      points_ = detail::makeAlignedArray<T>(Policy::kPointsPerParam * active_param_dim_ * param_dim_);
      counter_ = detail::makeAlignedArray<size_t>(detail::kCacheLine / sizeof(size_t));  // a cache line of its own
      partials_ = detail::makeAlignedArray<T>(partitioning_.num_partitions * partial_stride_);
      partials_capacity_ = partitioning_.num_partitions;
      args_ = detail::makeAlignedArray<detail::KernelArgs<T>>(1);
      ::new (static_cast<void*>(args_.get())) detail::KernelArgs<T>();
    }
    if (width > scratch_slices_) {
      scratch_ = detail::makeAlignedArray<unsigned char>(width * slice_layout_.stride,
                                                        detail::sliceAlignment(alignof(Model)));
      scratch_slices_ = width;
    }
  }

  /// Moves the given model into aligned storage: storage for all model_count_ models, or for the first alone when
  /// the cost runs stepped() (completeModels() builds the others).
  void buildModels() {
    model_slots_ = stepped() ? 1 : model_count_;
    model_storage_ = detail::makeAlignedArray<unsigned char>(model_slots_ * sizeof(Model),
                                                            detail::sliceAlignment(alignof(Model)));
    constructModels(reinterpret_cast<Model*>(model_storage_.get()));
  }

  /// Moves pending_model_ into slot 0 of `models`.
  void constructModels(Model* models) {
    ::new (static_cast<void*>(models)) Model(std::move(*pending_model_));
    models_built_ = 1;
    models_ = models;
    pending_model_.reset();
  }

  /// Copy-constructs the evaluation-point models of a numerical policy from slot 0, those not built
  /// yet. Their states are set before every launch, so the state slot 0 had does not matter. A
  /// constructor that throws leaves the ones built so far, and the next call goes on from there.
  void completeModels() {
    if constexpr (Policy::kPointsPerParam > 0) {
      if (models_built_ == model_count_) {
        return;
      }
      if (model_slots_ < model_count_) {
        // Storage for one model (a stepped() cost now used by a FusedCost): move it.
        auto storage = detail::makeAlignedArray<unsigned char>(model_count_ * sizeof(Model),
                                                              detail::sliceAlignment(alignof(Model)));
        Model* models = reinterpret_cast<Model*>(storage.get());
        ::new (static_cast<void*>(models)) Model(std::move(models_[0]));
        models_[0].~Model();
        models_ = models;
        model_storage_ = std::move(storage);
        model_slots_ = model_count_;
      }
      for (; models_built_ < model_count_; ++models_built_) {
        ::new (static_cast<void*>(models_ + models_built_)) Model(static_cast<const Model&>(models_[0]));
      }
    }
  }

  /// Whether this cost's own launches run the stepped path (runInline()): a numerical policy, a
  /// const-callable model and every element in one chunk.
  bool stepped() const {
    if constexpr (Policy::kPointsPerParam > 0 && Policy::kConstCallable) {
      return partitioning_.num_chunks == 1;
    } else {
      return false;
    }
  }

  void destroyModels() noexcept {
    if (models_ != nullptr) {
      for (size_t i = 0; i < models_built_; ++i) {
        models_[i].~Model();
      }
      models_ = nullptr;
    }
  }

  /// Evaluation points of a finite-difference policy: row `q` of the table is `x` stepped along
  /// active parameter `q / kPointsPerParam`, up for even `q % kPointsPerParam` and down for odd. With a
  /// manifold the step goes through its `plus`, on the tangent dimension; without, the entry is stepped
  /// in place.
  void fillPoints(const T* x) {
    constexpr size_t kPerParam = Policy::kPointsPerParam;
    const T step = Policy::step();
    T* table = points_.get();
    if (this->manifold() != nullptr) {
      std::vector<T> delta(tangent_dim_, T{0});
      for (size_t q = 0; q < kPerParam * active_param_dim_; ++q) {
        const size_t param = q / kPerParam;
        delta[param] = q % kPerParam == 0 ? step : -step;
        this->manifold()->plus(x, delta.data(), table + q * param_dim_);
        delta[param] = T{0};
      }
    } else {
      for (size_t q = 0; q < kPerParam * active_param_dim_; ++q) {
        const size_t param = q / kPerParam;
        T* row = table + q * param_dim_;
        std::copy_n(x, param_dim_, row);
        row[param] = x[param] + (q % kPerParam == 0 ? step : -step);
      }
    }
  }

  /// Everything the kernel reads, as of this evaluation: the loss descriptor is read now so that a
  /// scale changed between two evaluations is seen.
  detail::KernelArgs<T> makeArgs() const {
    detail::KernelArgs<T> args;
    args.x = x_.get();
    args.input = input_;
    args.observations = observations_;
    args.num_elements = num_elements_;
    args.input_dim = input_dim_;
    args.observation_dim = observation_dim_;
    args.params = active_param_dim_;
    args.param_dim = param_dim_;
    args.points = points_.get();
    if constexpr (Policy::kPointsPerParam > 0) {
      args.inv_span = Policy::invSpan();
    }
    args.model_count = model_count_;
    args.partitioning = partitioning_;
    args.weights = weights_.get();
    args.information_factors = information_factors_.get();
    if (loss_) {
      args.loss = loss_->descriptor();
      args.has_loss = true;
    }
    args.scratch = scratch_.get();
    args.slice = slice_layout_;
    args.partials = partials_.get();
    args.partial_stride = partial_stride_;
    args.counter = counter_.get();
    return args;
  }

  /// Number of work-items a launch of this cost uses on its own.
  template <bool kLinearSystem>
  size_t chunkedWidth() const {
    return kAccess == detail::ModelAccess::Exclusive
               ? 1
               : context_->launchWidth(partitioning_.num_partitions,
                                       kLinearSystem ? LaunchWork::LinearSystem : LaunchWork::CostOnly);
  }

  /// Everything an evaluation does on the host before its launch: buffers for `width` work-items, the
  /// model states at `x` (and at the evaluation points of a numerical linear system), the kernel
  /// arguments.
  /// `stepped`: the launch is a stepped runInline(), which needs no evaluation-point models.
  template <bool kLinearSystem>
  void prepare(const T* x, size_t width, bool stepped = false) {
    ensureState(width);
    if (!stepped) {
      completeModels();
    }

    std::copy_n(x, param_dim_, x_.get());
    models_[0].setState(x_.get());
    if constexpr (kLinearSystem && Policy::kPointsPerParam > 0) {
      fillPoints(x_.get());
      if (!stepped) {
        for (size_t i = 1; i < model_count_; ++i) {
          models_[i].setState(points_.get() + (i - 1) * param_dim_);
        }
      }
    }
    *args_ = makeArgs();
  }

  /// A launch of one work-item, on the calling thread with a plain claim counter. `stepped`: the
  /// work-item function runs over the one model of a stepped() cost, which goes from evaluation point
  /// to evaluation point (prepare() set its state to `x`).
  template <bool kLinearSystem>
  void runInline(bool stepped) {
    size_t next = 0;
    auto claim = [&next] { return next++; };
    detail::runOnCallingThread(*context_, [&] {
      if (stepped) {
        size_t current = 0;
        const detail::SteppedModels<Model, T> models{models_, x_.get(), points_.get(), param_dim_, &current};
        detail::runWorkItemInPlace<Policy, kLinearSystem>(*args_, models, 0, claim);
      } else {
        detail::runWorkItem<Policy, Model, T, kLinearSystem>(*args_, models_, 0, claim);
      }
    });
  }

  /// The chunked kernel's partials and claim counter, reset for a new launch.
  void resetChunkedPartials() {
    reduce_from_ = partials_.get();
    reduce_count_ = partitioning_.num_partitions;
    *counter_ = 0;
  }

  /// Sets the model's state to `x` and runs the one launch of an evaluation (wait() adds up its partials).
  /// One work-item runs inline on the calling thread; several run on the context's oneTBB arena and
  /// claim the partitions through the counter. Either way the same slots are written.
  template <bool kLinearSystem>
  void launch(const T* x) {
    const bool host_stepped = stepped();
    const size_t width = chunkedWidth<kLinearSystem>();
    prepare<kLinearSystem>(x, width, host_stepped);

    resetChunkedPartials();
    if (host_stepped || width == 1) {
      runInline<kLinearSystem>(host_stepped);
      return;
    }

    const detail::KernelArgs<T>* args = args_.get();
    Model* models = models_;
    detail::submit(*context_, width, [args, models](size_t group) {
      detail::runWorkItem<Policy, Model, T, kLinearSystem>(*args, models, group, [args] {
        return std::atomic_ref<size_t>(*args->counter).fetch_add(1, std::memory_order_relaxed);
      });
    });
  }

  // ─── Fused launches (FusedCost) ───────────────────────────────────────────────────────────────
  // A FusedCost runs the work-item function of several costs in one launch: each cost keeps
  // its own partitions, partials, claim counter and scratch, so its results are the ones its own launch
  // gives, bit for bit.
  template <class, class...>
  friend class FusedCost;

  /// What the fused launch needs of this cost.
  struct FusedRef {
    const detail::KernelArgs<T>* args = nullptr;
    Model* models = nullptr;
  };

  /// Prepares an evaluation that a FusedCost launches on `width` work-items (at least chunkedWidth()),
  /// and records it as pending with its outputs (`JTJ`/`JTb` unused for a cost-only evaluation). Returns
  /// false, with the outputs written, when there is nothing to launch (no elements).
  template <bool kLinearSystem>
  bool prepareFused(const T* x, size_t width, T* JTJ, T* JTb, T* cost) {
    wait();
    if (num_elements_ == 0) {
      if constexpr (kLinearSystem) {
        std::fill_n(JTJ, tangent_dim_ * tangent_dim_, T{0});
        std::fill_n(JTb, tangent_dim_, T{0});
      }
      *cost = T{0};
      return false;
    }
    prepare<kLinearSystem>(x, std::max(width, chunkedWidth<kLinearSystem>()));
    resetChunkedPartials();
    pending_ = Pending{kLinearSystem ? Pending::Kind::LinearSystem : Pending::Kind::Cost, JTJ, JTb, cost,
                       reduce_from_, reduce_count_};
    return true;
  }

  FusedRef fusedRef() const { return FusedRef{args_.get(), models_}; }

  /// The work of this cost in a fused launch, for work-item `worker`. A model that only
  /// one work-item may use runs on work-item 0 alone.
  template <bool kLinearSystem>
  static void runFused(const FusedRef& ref, size_t worker) {
    if constexpr (kAccess == detail::ModelAccess::Exclusive) {
      if (worker != 0) {
        return;
      }
    }
    size_t* counter = ref.args->counter;
    detail::runWorkItem<Policy, Model, T, kLinearSystem>(*ref.args, ref.models, worker, [counter] {
      return std::atomic_ref<size_t>(*counter).fetch_add(1, std::memory_order_relaxed);
    });
  }

  /// Completes the pending fused evaluation after its launch returned: adds the partials.
  void finishFused() {
    if (pending_.kind == Pending::Kind::None) {
      return;
    }
    const Pending pending = pending_;
    pending_.kind = Pending::Kind::None;
    reduce(pending);
  }

  /// The evaluation that was submitted and not yet completed by wait(), and where its results go.
  struct Pending {
    enum class Kind { None, Cost, LinearSystem };
    Kind kind = Kind::None;
    T* JTJ = nullptr;
    T* JTb = nullptr;
    T* cost = nullptr;
    const T* partials = nullptr;  ///< The partial results wait() adds up, in this order...
    size_t count = 0;             ///< ...and how many there are (one per partition).
  };

  /// Adds the partial results of a completed evaluation, in partition order, into its outputs.
  void reduce(const Pending& pending) {
    reducePartials(pending.partials, pending.count, partial_stride_, active_param_dim_, tangent_dim_,
                   pending.kind == Pending::Kind::LinearSystem, pending.JTJ, pending.JTb, pending.cost);
  }

  /// The reduction of `count` partial slots of `stride` entries, in slot order: the cost, and for a linear
  /// system JTb and both triangles of JTJ (`tangent_dim` square, zero outside the leading `params` block).
  static void reducePartials(const T* partials, size_t count, size_t stride, size_t params, size_t tangent_dim,
                             bool linear_system, T* JTJ, T* JTb, T* cost_out) {
    T cost{0};
    if (!linear_system) {
      for (size_t partition = 0; partition < count; ++partition) {
        cost += partials[partition * stride];
      }
      *cost_out = cost;
      return;
    }
    for (size_t i = 0; i < tangent_dim * tangent_dim; ++i) {
      JTJ[i] = T{0};
    }
    for (size_t i = 0; i < tangent_dim; ++i) {
      JTb[i] = T{0};
    }
    // Only the leading `params` block is populated; the trailing rows/columns (tangent dimensions this
    // cost does not depend on) stay zero. The partials hold the lower triangle.
    for (size_t partition = 0; partition < count; ++partition) {
      const T* partial = partials + partition * stride;
      cost += partial[0];
      for (size_t a = 0; a < params; ++a) {
        JTb[a] += partial[1 + a];
        for (size_t b = a; b < params; ++b) {
          JTJ[b + a * tangent_dim] += partial[1 + params + b + a * params];
        }
      }
    }
    for (size_t a = 0; a < params; ++a) {
      for (size_t b = a + 1; b < params; ++b) {
        JTJ[a + b * tangent_dim] = JTJ[b + a * tangent_dim];
      }
    }
    *cost_out = cost;
  }

  std::shared_ptr<ExecutionContext> context_;
  Pending pending_;
  std::optional<Model> pending_model_;  ///< Until the first evaluation moves it into its storage.
  const size_t model_count_;            ///< 1 + kPointsPerParam * active_param_dim.
  Model* models_ = nullptr;             ///< In model_storage_, constructed lazily, destroyed by ~ChunkedCost().
  size_t models_built_ = 0;
  size_t model_slots_ = 0;              ///< Models model_storage_ has room for.

  const T* input_;  ///< The caller's arrays.
  const T* observations_;

  detail::Partitioning partitioning_;  ///< Changes with setData().
  const size_t partial_stride_;
  detail::SliceLayout slice_layout_;   ///< Changes with setData().
  size_t partials_capacity_ = 0;       ///< Partition slots allocated in partials_.

  std::shared_ptr<LossFunction<T>> loss_;
  detail::AlignedPtr<T> weights_;               ///< One per element, or null.
  detail::AlignedPtr<T> information_factors_;   ///< observation_dim^2 per element (lower Cholesky factor), or null.

  // Buffers of the launches, allocated on first use.
  detail::AlignedPtr<unsigned char> model_storage_;
  detail::AlignedPtr<T> x_;
  detail::AlignedPtr<T> points_;  ///< kPointsPerParam * active_param_dim rows of param_dim entries.
  detail::AlignedPtr<size_t> counter_;
  detail::AlignedPtr<T> partials_;
  const T* reduce_from_ = nullptr;  ///< What the submitted evaluation's wait() adds up (see Pending).
  size_t reduce_count_ = 0;
  detail::AlignedPtr<detail::KernelArgs<T>> args_;
  detail::AlignedPtr<unsigned char> scratch_;
  size_t scratch_slices_ = 0;
};

}  // namespace moptim
