#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// A struct storing the total mass, momentum, and angular momentum of the
// deformable.
// MPM transfer should preserve mass and both momentums.
template <typename T>
struct MassAndMomentum {
  T total_mass = 0;
  Vector3<T> total_momentum{};
  Vector3<T> total_angular_momentum{};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
