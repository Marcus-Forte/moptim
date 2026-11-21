#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "moptim/AnalyticalCost.hh"
#include "moptim/NumericalCostCentral.hh"
#include "moptim/NumericalCostForwardEuler.hh"

namespace tu_models {

/// Fits y = k * x0 * in with a known `k` through every cost type: the linear system must be that of
/// the residual of this translation unit's model, whichever other models of the same name the
/// executable holds.
template <class Model>
void check(double k) {
  std::vector<double> in, obs;
  for (int i = 0; i < 3000; ++i) {
    in.push_back(0.001 * i - 1.0);
    obs.push_back(0.5 + 0.0003 * i);
  }
  const double x = 0.7;
  double jtj_ref = 0.0, jtb_ref = 0.0, cost_ref = 0.0;
  for (size_t i = 0; i < in.size(); ++i) {
    const double r = obs[i] - k * x * in[i];
    const double j = -k * in[i];
    jtj_ref += j * j;
    jtb_ref += j * r;
    cost_ref += r * r;
  }

  const auto verify = [&](auto& cost, double tolerance) {
    double jtj = 0.0, jtb = 0.0, value = 0.0;
    cost.computeLinearSystem(&x, &jtj, &jtb, value);
    EXPECT_NEAR(jtj, jtj_ref, tolerance * jtj_ref);
    EXPECT_NEAR(jtb, jtb_ref, tolerance * std::abs(jtb_ref));
    EXPECT_NEAR(value, cost_ref, 1e-12 * cost_ref);
    EXPECT_NEAR(cost.computeCost(&x), cost_ref, 1e-12 * cost_ref);
  };
  moptim::AnalyticalCost<Model, double> analytical(in.data(), obs.data(), in.size(), 1, 1, 1);
  verify(analytical, 1e-12);
  moptim::NumericalCostForwardEuler<Model, double> forward(in.data(), obs.data(), in.size(), 1, 1, 1);
  verify(forward, 1e-5);
  moptim::NumericalCostCentral<Model, double> central(in.data(), obs.data(), in.size(), 1, 1, 1);
  verify(central, 1e-5);
}

}  // namespace tu_models
