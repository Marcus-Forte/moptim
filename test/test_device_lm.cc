// DeviceLevenbergMarquardt: a whole optimize() in one launch, against LevenbergMarquardt with speculative
// linear systems (which it follows step for step, up to rounding).
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "manifold_models.hh"
#include "moptim/AnalyticalCost.hh"
#include "moptim/DeviceLevenbergMarquardt.hh"
#include "moptim/ExecutionContext.hh"
#include "moptim/FusedCost.hh"
#include "moptim/LevenbergMarquardt.hh"
#include "moptim/NumericalCostCentral.hh"

using namespace moptim;

namespace {

constexpr size_t kParams = 4;

struct CurveModel {
  void setState(const double* /*x*/) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const {
    res[0] = obs[0] - (x[0] * in[0] + std::sin(x[1] * in[1]) + x[2]);
    res[1] = obs[1] - (x[3] * in[0] * in[1] + x[0] * x[1]);
  }
  void jacobian(const double* x, const double* in, const double* /*obs*/, double* jac) const {
    jac[0] = -in[0];
    jac[1] = -in[1] * std::cos(x[1] * in[1]);
    jac[2] = -1.0;
    jac[3] = 0.0;
    jac[4] = -x[1];
    jac[5] = -x[0];
    jac[6] = 0.0;
    jac[7] = -in[0] * in[1];
  }
};

struct Data {
  std::vector<double> input;
  std::vector<double> observations;
};

Data makeData(size_t n, const double* truth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::normal_distribution<double> noise(0.0, 0.05);
  Data data;
  CurveModel model;
  for (size_t e = 0; e < n; ++e) {
    const double in[2] = {value(rng), value(rng)};
    double obs[2];
    model.residual(truth, in, std::array<double, 2>{0.0, 0.0}.data(), obs);
    data.input.push_back(in[0]);
    data.input.push_back(in[1]);
    data.observations.push_back(-obs[0] + noise(rng));
    data.observations.push_back(-obs[1] + noise(rng));
  }
  return data;
}

std::shared_ptr<ExecutionContext> makeContext() {
  return std::make_shared<ExecutionContext>(defaultBackend());
}

const double kTruth[kParams] = {0.5, -0.4, 0.1, 0.9};
const double kStart[kParams] = {0.3, -0.7, 0.2, 1.1};

using Curve = AnalyticalCost<CurveModel, double>;
using CurveCentral = NumericalCostCentral<CurveModel, double>;
using Euclid = EuclideanSpace<double, kParams>;

/// SE(3) on the device: manifold_models::SE3Manifold::plus as a static function.
struct SE3Device {
  static constexpr size_t kAmbientDim = 12;
  static constexpr size_t kTangentDim = 6;
  static void plus(const double* x, const double* delta, double* x_plus) {
    Eigen::Matrix3d dR;
    Eigen::Vector3d dt;
    manifold_models::se3Exp(delta, dR, dt);
    const Eigen::Map<const Eigen::Matrix3d> R(x);
    const Eigen::Map<const Eigen::Vector3d> t(x + 9);
    Eigen::Map<Eigen::Matrix3d>{x_plus} = dR * R;
    Eigen::Map<Eigen::Vector3d>{x_plus + 9} = dR * t + dt;
  }
};

}  // namespace

TEST(DeviceLevenbergMarquardt, FollowsTheHostOptimizerInOneLaunch) {
  const Data data = makeData(3000, kTruth, 61);
  const auto context = makeContext();
  const auto curve = std::make_shared<Curve>(data.input.data(), data.observations.data(), 3000, 2, 2, kParams,
                                             CurveModel{}, 0, context);
  const auto central = std::make_shared<CurveCentral>(data.input.data(), data.observations.data(), 700, 2, 2,
                                                      kParams, CurveModel{}, 0, context);
  central->setLossFunction(std::make_shared<HuberLoss<double>>(0.3));

  for (const size_t iterations : {size_t{1}, size_t{3}, size_t{15}}) {
    SCOPED_TRACE(testing::Message() << "iterations=" << iterations);
    LevenbergMarquardt<double> host(kParams);
    host.setSpeculativeLinearSystems(true);
    host.setMaxIterations(iterations);
    host.addCost(curve);
    host.addCost(central);
    std::vector<double> x_host(kStart, kStart + kParams);
    const Result<double> host_result = host.optimize(x_host.data());

    DeviceLevenbergMarquardt<double, Euclid, Curve, CurveCentral> device(curve, central);
    device.setMaxIterations(iterations);
    std::vector<double> x_device(kStart, kStart + kParams);
    const uint64_t before = context->launches();
    const Result<double> device_result = device.optimize(x_device.data());
    EXPECT_EQ(context->launches() - before, 1u);

    EXPECT_EQ(device_result.status, host_result.status);
    EXPECT_EQ(device_result.iterations, host_result.iterations);
    EXPECT_NEAR(device_result.final_cost, host_result.final_cost, 1e-9 * host_result.final_cost);
    for (size_t i = 0; i < kParams; ++i) {
      EXPECT_NEAR(x_device[i], x_host[i], 1e-9);
    }
    // Run to run, bit for bit.
    std::vector<double> x_again(kStart, kStart + kParams);
    device.optimize(x_again.data());
    EXPECT_EQ(std::memcmp(x_again.data(), x_device.data(), sizeof(double) * kParams), 0);
  }
}

TEST(DeviceLevenbergMarquardt, WorksOnSE3WithReusedCosts) {
  std::mt19937 rng(67);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const size_t n = 1200;
  Eigen::Matrix<double, 6, 1> truth_delta;
  truth_delta << 0.02, -0.01, 0.015, 0.01, -0.02, 0.03;
  const Eigen::Matrix<double, 12, 1> truth = manifold_models::poseFromTangent(truth_delta);
  const Eigen::Map<const Eigen::Matrix3d> R(truth.data());
  const Eigen::Map<const Eigen::Vector3d> t(truth.data() + 9);
  std::vector<double> planes, offsets;
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d p(value(rng), value(rng), value(rng));
    const Eigen::Vector3d normal = Eigen::Vector3d(value(rng), value(rng), value(rng)).normalized();
    planes.insert(planes.end(), {p.x(), p.y(), p.z(), normal.x(), normal.y(), normal.z()});
    offsets.push_back(normal.dot(R * p + t));
  }
  const auto manifold = std::make_shared<manifold_models::SE3Manifold>();
  const auto context = makeContext();
  using Plane = AnalyticalCost<manifold_models::PointToPlane, double>;
  const auto plane = std::make_shared<Plane>(planes.data(), offsets.data(), n, 6, 1, manifold,
                                             manifold_models::PointToPlane{}, 0, context);
  DeviceLevenbergMarquardt<double, SE3Device, Plane> device(plane);
  device.setMaxIterations(5);
  LevenbergMarquardt<double> host(manifold);
  host.setSpeculativeLinearSystems(true);
  host.setMaxIterations(5);
  host.addCost(plane);

  for (const size_t elements : {n, n / 3}) {
    SCOPED_TRACE(testing::Message() << "elements=" << elements);
    plane->setData(planes.data(), offsets.data(), elements);
    const Eigen::Matrix<double, 12, 1> start = manifold_models::poseFromTangent(Eigen::Matrix<double, 6, 1>::Zero());
    Eigen::Matrix<double, 12, 1> x_host = start, x_device = start;
    const auto host_result = host.optimize(x_host.data());
    const auto device_result = device.optimize(x_device.data());
    // At convergence the status (SMALL_DELTA or MAX_ITERATIONS_REACHED) is decided by whether a last,
    // tiny step raises the cost at the rounding level, which the device optimizer's own Cholesky can
    // decide differently: only the result is compared.
    (void)device_result;
    (void)host_result;
    EXPECT_LT((x_device - x_host).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((x_device - truth).cwiseAbs().maxCoeff(), 1e-6);
  }
}

TEST(DeviceLevenbergMarquardt, RejectsCostsOfAnotherContextOrSize) {
  const Data data = makeData(100, kTruth, 71);
  const auto a = std::make_shared<Curve>(data.input.data(), data.observations.data(), 100, 2, 2, kParams,
                                         CurveModel{}, 0, makeContext());
  const auto b = std::make_shared<Curve>(data.input.data(), data.observations.data(), 100, 2, 2, kParams,
                                         CurveModel{}, 0, makeContext());
  EXPECT_THROW((DeviceLevenbergMarquardt<double, Euclid, Curve, Curve>(a, b)), std::invalid_argument);
  EXPECT_THROW((DeviceLevenbergMarquardt<double, EuclideanSpace<double, 3>, Curve>(a)), std::invalid_argument);
}
