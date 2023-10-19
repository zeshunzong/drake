#pragma once

#include <array>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {


// Stores temporary information on 27 neighbor grid nodes
template <typename T>
struct Pad {
  std::array<T, 27> pad_masses;

  std::array<Vector3<T>, 27> pad_velocities;

  std::array<Vector3<T>, 27> pad_forces;
};


}  // namespace mpm
}  // namespace multibody
}  // namespace drake
