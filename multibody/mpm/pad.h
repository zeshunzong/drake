#pragma once

#include <array>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A pad stores the temporary data that will contribute to GridData.
 * All particles who share the same batch index will first write to the same
 * pad.
 */
template <typename T>
struct Pad {
  std::array<T, 27> pad_masses;

  std::array<Vector3<T>, 27> pad_velocities;

  std::array<Vector3<T>, 27> pad_forces;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
