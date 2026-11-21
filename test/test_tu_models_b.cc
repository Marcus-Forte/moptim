// Two translation units (this and test_tu_models_a.cc) each declare their own `Model` in an anonymous
// namespace, with different residuals, and both end up in test_moptim. Their kernels must not be
// confused with each other.
#include "tu_models_common.hh"

namespace {

// residual = obs - 5 * x[0] * in
struct Model {
  void setState(const double*) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const { res[0] = obs[0] - 5.0 * x[0] * in[0]; }
  void jacobian(const double*, const double* in, const double*, double* jac) const { jac[0] = -5.0 * in[0]; }
};

}  // namespace

void checkTuModelsB() { tu_models::check<Model>(5.0); }
