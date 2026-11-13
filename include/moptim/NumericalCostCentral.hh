#pragma once

#include "moptim/NumericalCost.hh"

namespace moptim {

/// Cost differentiated with central (symmetric) finite differences.
template <class Model, class T>
using NumericalCostCentral = NumericalCost<Model, T, DifferenceScheme::Central>;

}  // namespace moptim
