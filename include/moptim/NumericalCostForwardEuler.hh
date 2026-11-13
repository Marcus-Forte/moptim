#pragma once

#include "moptim/NumericalCost.hh"

namespace moptim {

/// Cost differentiated with forward (one-sided) finite differences.
template <class Model, class T>
using NumericalCostForwardEuler = NumericalCost<Model, T, DifferenceScheme::Forward>;

}  // namespace moptim
