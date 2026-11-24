// Two translation units (this and test_tu_models_b.cc) each declare their own `Model` in an anonymous
// namespace, with different residuals, and both end up in test_moptim. Their costs (and the
// instantiations of the templates they share) must not be confused with each other.
#include "tu_models_common.hh"

namespace {

// residual = obs - 2 * x[0] * in
struct Model {
  void setState(const double*) {}
  void residual(const double* x, const double* in, const double* obs, double* res) const { res[0] = obs[0] - 2.0 * x[0] * in[0]; }
  void jacobian(const double*, const double* in, const double*, double* jac) const { jac[0] = -2.0 * in[0]; }
};

}  // namespace

void checkTuModelsA() { tu_models::check<Model>(2.0); }
