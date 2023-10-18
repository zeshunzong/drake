#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// Stores the sum of mass, momentum and angular momentum of the grid/particles.
// The angular momentum is about the origin in the world frame.
// This is used in validation of conservation in MPM transfers.
template <typename T>
struct TotalMassMomentum {
  T total_mass;
  Vector3<T> total_momentum;
  Vector3<T> total_angular_momentum;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
