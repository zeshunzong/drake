#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace mathutils {

// Return (i, j, k)th entry of the third order permutation tensor
double LeviCivita(int i, int j, int k);

// Calculate A:ε
Vector3<double> ContractionWithLeviCivita(const Matrix3<double>& A);

}  // namespace mathutils
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
