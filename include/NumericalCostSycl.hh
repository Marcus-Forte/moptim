#pragma once

#include "ICost.hh"
#include "sycl/sycl.hpp"

namespace {
const double g_step = std::sqrt(std::numeric_limits<double>::epsilon());

enum class DifferentiationMethod { BACKWARD_EULER = 0, CENTRAL = 1 };
}  // namespace

template <class InputT, class OutputT, class Model,
          ::DifferentiationMethod MethodT = ::DifferentiationMethod::BACKWARD_EULER>
class NumericalCostSycl : public ICost {
  static constexpr size_t OutputDim = sizeof(OutputT) / sizeof(double);
  using ResidualVectorT = Eigen::Vector<double, OutputDim>;

 public:
  NumericalCostSycl(const NumericalCostSycl&) = delete;
  NumericalCostSycl(const std::vector<InputT>* input, const std::vector<OutputT>* observations)
      : input_{input}, observations_{observations}, queue_(sycl::default_selector_v) {
    input_sycl_ = sycl::malloc_device<InputT>(input->size(), queue_);
    observations_sycl_ = sycl::malloc_device<OutputT>(observations->size(), queue_);

    /// \todo Copy data to GPU now?
    queue_.copy<InputT>(input->data(), input_sycl_, input->size()).wait();
    queue_.copy<OutputT>(observations->data(), observations_sycl_, observations->size()).wait();
  }

  virtual ~NumericalCostSycl() {
    sycl::free(input_sycl_, queue_);
    sycl::free(observations_sycl_, queue_);
  }

  double computeCost(const Eigen::VectorXd& x) override {
    auto* cost_reduction = sycl::malloc_device<double>(1, queue_);

    auto input_capture = input_sycl_;
    auto observations_capture = observations_sycl_;

    Model model(x);
    auto* model_sycl = sycl::malloc_device<Model>(1, queue_);
    queue_.copy<Model>(&model, model_sycl, 1).wait();

    const auto func = [](double input, double measurement) { return measurement - 0.1 * input / (0.1 + input); };

    queue_.single_task([=]() { *cost_reduction = 0.0F; }).wait();

    queue_
        .submit([&](sycl::handler& cgh) {
          auto sum = sycl::reduction(cost_reduction, 0.0, sycl::plus<double>{});

          cgh.parallel_for(sycl::range<1>(input_->size()), sum, [=](sycl::id<1> id, auto& reduction) {
            /// \todo Model objets (model_sycl) won't work in GPU
            auto error = func(input_capture[id], observations_capture[id]);

            // Following Eigen construct works in NVIDIA GPU! :)
            Eigen::Map<const ResidualVectorT> residual_map(reinterpret_cast<const double*>(&error));
            // error = error + 1.0;
            // *cost_reduction = error;
            reduction += residual_map.squaredNorm();
          });
        })
        .wait();

    double result;
    queue_.copy<double>(cost_reduction, &result, 1).wait();
    sycl::free(cost_reduction, queue_);
    return result;
  }

  SolveRhs computeLinearSystem(const Eigen::VectorXd& x) override { throw std::runtime_error("Unimplemented"); }

 private:
  const std::vector<InputT>* input_;
  const std::vector<OutputT>* observations_;

  InputT* input_sycl_;
  OutputT* observations_sycl_;
  //   OutputT* output_sycl_;
  sycl::queue queue_;
};