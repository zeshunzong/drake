#pragma once

#include <array>
#include <memory>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct Pad {
  std::array<T, 27> masses{};
  std::array<Vector3<T>, 27> momentums{};
  std::array<Vector3<T>, 27> forces{};
  Vector3<int> base_node{};

  // the 27 local nodes (3by3by3) of a cube are stored lexicographically as,
  // viewing the center node as {0,0,0}

  // @pre a, b, c ∈ {−1,0,1}
  const T& GetMassAt(int a, int b, int c) const {
    int id_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
    return masses[id_local];
  }

  // @pre a, b, c ∈ {−1,0,1}
  const Vector3<T>& GetMomentumAt(int a, int b, int c) const {
    int id_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
    return momentums[id_local];
  }

  // @pre a, b, c ∈ {−1,0,1}
  const Vector3<T>& GetForceAt(int a, int b, int c) const {
    int id_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
    return forces[id_local];
  }
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
