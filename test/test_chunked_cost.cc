#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/LossFunction.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"

using namespace moptim;

namespace {

constexpr size_t kInputDim = 2;
constexpr size_t kObservationDim = 2;
constexpr size_t kParamDim = 4;

// A small nonlinear model with two residuals per element and four parameters.
struct CurveModel {
  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* input, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] * input[0] + std::sin(x[1] * input[1]) + x[2]);
    res[1] = obs[1] - (x[3] * input[0] * input[1] + x[0] * x[1]);
  }

  // jac[j + kParamDim * i] = d residual[i] / d x[j]
  void jacobian(const double* x, const double* input, const double* /*obs*/, double* jac) const {
    jac[0 + kParamDim * 0] = -input[0];
    jac[1 + kParamDim * 0] = -input[1] * std::cos(x[1] * input[1]);
    jac[2 + kParamDim * 0] = -1.0;
    jac[3 + kParamDim * 0] = 0.0;

    jac[0 + kParamDim * 1] = -x[1];
    jac[1 + kParamDim * 1] = -x[0];
    jac[2 + kParamDim * 1] = 0.0;
    jac[3 + kParamDim * 1] = -input[0] * input[1];
  }
};

struct Dataset {
  std::vector<double> input;
  std::vector<double> observations;
  std::vector<double> weights;
  std::vector<Eigen::MatrixXd> information;

  size_t size() const { return weights.size(); }
};

Dataset makeDataset(size_t num_elements) {
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> weight(0.25, 4.0);

  Dataset data;
  for (size_t e = 0; e < num_elements; ++e) {
    for (size_t i = 0; i < kInputDim; ++i) data.input.push_back(value(rng));
    for (size_t i = 0; i < kObservationDim; ++i) data.observations.push_back(value(rng));
    data.weights.push_back(weight(rng));

    Eigen::Matrix2d factor;
    factor << value(rng), value(rng), value(rng), value(rng);
    data.information.emplace_back(factor * factor.transpose() + Eigen::Matrix2d::Identity());
  }
  return data;
}

Eigen::VectorXd linearizationPoint() {
  Eigen::VectorXd x(kParamDim);
  x << 0.3, -0.7, 0.2, 1.1;
  return x;
}

enum class Noise { None, Weights, Information };

struct LinearSystem {
  Eigen::MatrixXd jtj = Eigen::MatrixXd::Zero(kParamDim, kParamDim);
  Eigen::VectorXd jtb = Eigen::VectorXd::Zero(kParamDim);
  double cost = 0.0;
};

/// The linear system written element by element with plain dense algebra, straight from its
/// definition: J^T (rho' * Omega) J, J^T (rho' * Omega) r and sum rho(r^T Omega r).
LinearSystem referenceSystem(const Dataset& data, const Eigen::VectorXd& x, Noise noise,
                             const LossFunction<double>* loss) {
  CurveModel model;
  LinearSystem system;
  for (size_t e = 0; e < data.size(); ++e) {
    const double* input = &data.input[e * kInputDim];
    const double* observation = &data.observations[e * kObservationDim];

    Eigen::Vector2d residual;
    model.residual(x.data(), input, observation, residual.data());
    Eigen::Matrix<double, kParamDim, kObservationDim> jacobian_transposed;
    model.jacobian(x.data(), input, observation, jacobian_transposed.data());
    const Eigen::Matrix<double, kObservationDim, kParamDim> jacobian = jacobian_transposed.transpose();

    Eigen::Matrix2d omega = Eigen::Matrix2d::Identity();
    if (noise == Noise::Weights) omega *= data.weights[e];
    if (noise == Noise::Information) omega = data.information[e];

    const double squared_norm = residual.dot(omega * residual);
    double rho[3] = {squared_norm, 1.0, 0.0};
    if (loss != nullptr) loss->evaluate(squared_norm, rho);

    system.jtj += rho[1] * jacobian.transpose() * omega * jacobian;
    system.jtb += rho[1] * jacobian.transpose() * omega * residual;
    system.cost += rho[0];
  }
  return system;
}

template <class Cost>
void configure(Cost& cost, const Dataset& data, Noise noise, const std::shared_ptr<LossFunction<double>>& loss) {
  if (noise == Noise::Weights) cost.setWeights(data.weights);
  if (noise == Noise::Information) cost.setInformation(data.information);
  cost.setLossFunction(loss);
}

template <class Cost>
LinearSystem linearSystem(Cost& cost, const Eigen::VectorXd& x) {
  LinearSystem system;
  // Start from garbage: computeLinearSystem() must overwrite, not accumulate.
  system.jtj.setConstant(123.0);
  system.jtb.setConstant(-456.0);
  system.cost = 789.0;
  cost.computeLinearSystem(x.data(), system.jtj.data(), system.jtb.data(), system.cost);
  return system;
}

void expectSameSystem(const LinearSystem& actual, const LinearSystem& expected, double tolerance) {
  EXPECT_LE((actual.jtj - expected.jtj).norm(), tolerance * (1.0 + expected.jtj.norm())) << "JTJ\n"
                                                                                         << actual.jtj << "\nexpected\n"
                                                                                         << expected.jtj;
  EXPECT_LE((actual.jtb - expected.jtb).norm(), tolerance * (1.0 + expected.jtb.norm()))
      << "JTb " << actual.jtb.transpose() << "\nexpected " << expected.jtb.transpose();
  EXPECT_NEAR(actual.cost, expected.cost, tolerance * (1.0 + std::abs(expected.cost)));
}

// Element counts on both sides of the chunk size (204 elements for this model): none, one, one
// chunk minus/plus one element, an exact number of chunks and several chunks with a partial last one.
const size_t kElementCounts[] = {0, 1, 203, 204, 205, 408, 1000};

const Noise kNoiseModels[] = {Noise::None, Noise::Weights, Noise::Information};

const char* name(Noise noise) {
  switch (noise) {
    case Noise::None:
      return "none";
    case Noise::Weights:
      return "weights";
    case Noise::Information:
      return "information";
  }
  return "?";
}

}  // namespace

TEST(ChunkedCost, AnalyticalMatchesDenseReference) {
  const Eigen::VectorXd x = linearizationPoint();

  for (const size_t num_elements : kElementCounts) {
    const Dataset data = makeDataset(num_elements);
    for (const Noise noise : kNoiseModels) {
      for (const bool robust : {false, true}) {
        SCOPED_TRACE(testing::Message() << "elements=" << num_elements << " noise=" << name(noise)
                                        << " robust=" << robust);
        const std::shared_ptr<LossFunction<double>> loss = robust ? std::make_shared<HuberLoss<double>>(1.5) : nullptr;

        AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), num_elements, kInputDim,
                                                kObservationDim, kParamDim);
        configure(cost, data, noise, loss);

        const LinearSystem expected = referenceSystem(data, x, noise, loss.get());
        const LinearSystem actual = linearSystem(cost, x);
        expectSameSystem(actual, expected, 1e-12);
        // Both triangles are written, and they mirror each other exactly.
        EXPECT_EQ(actual.jtj, actual.jtj.transpose());
        EXPECT_NEAR(cost.computeCost(x.data()), expected.cost, 1e-12 * (1.0 + expected.cost));
      }
    }
  }
}

// Element counts that span many chunks and several partitions of the kernel (a partition is four
// chunks, 816 elements for this model), checked against the same dense reference.
TEST(ChunkedCost, AnalyticalMatchesDenseReferenceOverManyPartitions) {
  const Eigen::VectorXd x = linearizationPoint();

  for (const size_t num_elements : {size_t{20000}, size_t{50001}}) {
    const Dataset data = makeDataset(num_elements);
    for (const Noise noise : kNoiseModels) {
      for (const bool robust : {false, true}) {
        SCOPED_TRACE(testing::Message() << "elements=" << num_elements << " noise=" << name(noise)
                                        << " robust=" << robust);
        const std::shared_ptr<LossFunction<double>> loss = robust ? std::make_shared<HuberLoss<double>>(1.5) : nullptr;

        AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), num_elements, kInputDim,
                                                kObservationDim, kParamDim);
        configure(cost, data, noise, loss);

        const LinearSystem expected = referenceSystem(data, x, noise, loss.get());
        const LinearSystem actual = linearSystem(cost, x);
        expectSameSystem(actual, expected, 1e-12);
        EXPECT_EQ(actual.jtj, actual.jtj.transpose());
        EXPECT_NEAR(cost.computeCost(x.data()), expected.cost, 1e-12 * (1.0 + expected.cost));
        EXPECT_EQ(actual.cost, cost.computeCost(x.data()));
      }
    }
  }
}

TEST(ChunkedCost, NumericalMatchesDenseReference) {
  const Eigen::VectorXd x = linearizationPoint();

  for (const size_t num_elements : kElementCounts) {
    const Dataset data = makeDataset(num_elements);
    for (const Noise noise : kNoiseModels) {
      for (const bool robust : {false, true}) {
        SCOPED_TRACE(testing::Message() << "elements=" << num_elements << " noise=" << name(noise)
                                        << " robust=" << robust);
        // Cauchy is smooth, so the finite-difference error of the jacobian carries over smoothly.
        const std::shared_ptr<LossFunction<double>> loss = robust ? std::make_shared<CauchyLoss<double>>(1.5) : nullptr;
        const LinearSystem expected = referenceSystem(data, x, noise, loss.get());

        NumericalCostForwardEuler<CurveModel, double> forward(data.input.data(), data.observations.data(), num_elements,
                                                              kInputDim, kObservationDim, kParamDim);
        configure(forward, data, noise, loss);
        expectSameSystem(linearSystem(forward, x), expected, 1e-5);
        EXPECT_NEAR(forward.computeCost(x.data()), expected.cost, 1e-12 * (1.0 + expected.cost));

        NumericalCostCentral<CurveModel, double> central(data.input.data(), data.observations.data(), num_elements,
                                                         kInputDim, kObservationDim, kParamDim);
        configure(central, data, noise, loss);
        expectSameSystem(linearSystem(central, x), expected, 1e-6);
        EXPECT_NEAR(central.computeCost(x.data()), expected.cost, 1e-12 * (1.0 + expected.cost));
      }
    }
  }
}

// The optimizers compare the cost of a linear system with the cost of a trial point, and report
// the former as the final cost when no trial is accepted: both must be the same number.
TEST(ChunkedCost, LinearSystemCostEqualsComputeCost) {
  const Eigen::VectorXd x = linearizationPoint();
  const Dataset data = makeDataset(1000);

  for (const Noise noise : kNoiseModels) {
    for (const bool robust : {false, true}) {
      SCOPED_TRACE(testing::Message() << "noise=" << name(noise) << " robust=" << robust);
      const std::shared_ptr<LossFunction<double>> loss = robust ? std::make_shared<HuberLoss<double>>(1.5) : nullptr;

      AnalyticalCost<CurveModel, double> analytical(data.input.data(), data.observations.data(), data.size(), kInputDim,
                                                    kObservationDim, kParamDim);
      configure(analytical, data, noise, loss);
      EXPECT_DOUBLE_EQ(linearSystem(analytical, x).cost, analytical.computeCost(x.data()));

      NumericalCostCentral<CurveModel, double> numerical(data.input.data(), data.observations.data(), data.size(),
                                                         kInputDim, kObservationDim, kParamDim);
      configure(numerical, data, noise, loss);
      EXPECT_DOUBLE_EQ(linearSystem(numerical, x).cost, numerical.computeCost(x.data()));
    }
  }
}

// Weights and information matrices replace each other, and an empty vector removes either.
TEST(ChunkedCost, NoiseModelsReplaceEachOther) {
  const Eigen::VectorXd x = linearizationPoint();
  const Dataset data = makeDataset(300);

  AnalyticalCost<CurveModel, double> cost(data.input.data(), data.observations.data(), data.size(), kInputDim,
                                          kObservationDim, kParamDim);
  const LinearSystem plain = referenceSystem(data, x, Noise::None, nullptr);
  const LinearSystem weighted = referenceSystem(data, x, Noise::Weights, nullptr);
  const LinearSystem informed = referenceSystem(data, x, Noise::Information, nullptr);

  cost.setWeights(data.weights);
  expectSameSystem(linearSystem(cost, x), weighted, 1e-12);

  cost.setInformation(data.information);
  expectSameSystem(linearSystem(cost, x), informed, 1e-12);

  cost.setWeights(data.weights);
  expectSameSystem(linearSystem(cost, x), weighted, 1e-12);

  cost.setWeights({});
  expectSameSystem(linearSystem(cost, x), plain, 1e-12);

  cost.setInformation(data.information);
  cost.setInformation({});
  expectSameSystem(linearSystem(cost, x), plain, 1e-12);
}

// A parameter vector wider than what the model depends on, over several chunks: the active block
// must match the reference and everything outside it must be exactly zero.
TEST(ChunkedCost, ActiveBlockOverSeveralChunks) {
  constexpr size_t kFullDim = 7;
  const Dataset data = makeDataset(1000);

  Eigen::VectorXd x = Eigen::VectorXd::Constant(kFullDim, 5.0);
  x.head(kParamDim) = linearizationPoint();
  const LinearSystem expected = referenceSystem(data, linearizationPoint(), Noise::Weights, nullptr);

  const auto check = [&](auto& cost) {
    cost.setWeights(data.weights);
    Eigen::MatrixXd jtj = Eigen::MatrixXd::Constant(kFullDim, kFullDim, 123.0);
    Eigen::VectorXd jtb = Eigen::VectorXd::Constant(kFullDim, 123.0);
    double value = 0.0;
    cost.computeLinearSystem(x.data(), jtj.data(), jtb.data(), value);

    LinearSystem active;
    active.jtj = jtj.topLeftCorner(kParamDim, kParamDim);
    active.jtb = jtb.head(kParamDim);
    active.cost = value;
    expectSameSystem(active, expected, 1e-6);

    EXPECT_TRUE((jtj.bottomRows(kFullDim - kParamDim).array() == 0.0).all());
    EXPECT_TRUE((jtj.rightCols(kFullDim - kParamDim).array() == 0.0).all());
    EXPECT_TRUE((jtb.tail(kFullDim - kParamDim).array() == 0.0).all());
  };

  AnalyticalCost<CurveModel, double> analytical(data.input.data(), data.observations.data(), data.size(), kInputDim,
                                                kObservationDim, kFullDim, CurveModel{}, kParamDim);
  check(analytical);
  NumericalCostCentral<CurveModel, double> central(data.input.data(), data.observations.data(), data.size(), kInputDim,
                                                   kObservationDim, kFullDim, CurveModel{}, kParamDim);
  check(central);
}

namespace {

// Mirrors how a consumer wraps external state: the model keeps a reference, so it is neither
// default-constructible nor assignable, and it ignores setState().
class LinearFactor {
 public:
  static constexpr int kDim = 9;
  using Matrix = Eigen::Matrix<double, kDim, kDim>;
  using Vector = Eigen::Matrix<double, kDim, 1>;

  explicit LinearFactor(const Matrix& design) : design_(design) {}

  void setState(const double* /*x*/) {}

  void residual(const double* x, const double* /*input*/, const double* obs, double* res) const {
    Eigen::Map<Vector>{res} = design_ * Eigen::Map<const Vector>{x} - Eigen::Map<const Vector>{obs};
  }

 private:
  const Matrix& design_;
};

}  // namespace

// One element whose residual block is wider than the fused kernels handle, differentiated through
// copies of a model that cannot be assigned.
TEST(ChunkedCost, SingleWideElementWithReferenceHoldingModel) {
  constexpr int kDim = LinearFactor::kDim;
  LinearFactor::Matrix design = LinearFactor::Matrix::Random();
  design.diagonal().array() += 5.0;
  const LinearFactor::Vector target = LinearFactor::Vector::Random();
  const LinearFactor::Vector unused_input = LinearFactor::Vector::Zero();
  const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(kDim, -1.0, 1.0);

  const LinearFactor::Vector residual = design * x - target;
  const Eigen::MatrixXd expected_jtj = design.transpose() * design;
  const Eigen::VectorXd expected_jtb = design.transpose() * residual;

  for (const bool central : {false, true}) {
    SCOPED_TRACE(testing::Message() << "central=" << central);
    Eigen::MatrixXd jtj(kDim, kDim);
    Eigen::VectorXd jtb(kDim);
    double cost = 0.0;
    if (central) {
      NumericalCostCentral<LinearFactor, double> numerical(unused_input.data(), target.data(), 1, kDim, kDim, kDim,
                                                           LinearFactor(design));
      numerical.computeLinearSystem(x.data(), jtj.data(), jtb.data(), cost);
    } else {
      NumericalCostForwardEuler<LinearFactor, double> numerical(unused_input.data(), target.data(), 1, kDim, kDim, kDim,
                                                                LinearFactor(design));
      numerical.computeLinearSystem(x.data(), jtj.data(), jtb.data(), cost);
    }

    EXPECT_LE((jtj - expected_jtj).norm(), 1e-5 * expected_jtj.norm());
    EXPECT_LE((jtb - expected_jtb).norm(), 1e-5 * expected_jtb.norm());
    EXPECT_NEAR(cost, residual.squaredNorm(), 1e-10);
  }
}
