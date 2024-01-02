#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct P2gPad {
  std::array<T, 27> masses{};
  std::array<Vector3<T>, 27> momentums{};
  std::array<Vector3<T>, 27> forces{};
  Vector3<int> base_node{};

  // the 27 local nodes (3by3by3) of a cube are stored lexicographically as,
  // viewing the center node as {0,0,0}

  // initialize to zero
  void Reset() {
    std::fill(momentums.begin(), momentums.end(), Vector3<T>::Zero());
    std::fill(forces.begin(), forces.end(), Vector3<T>::Zero());
    std::fill(masses.begin(), masses.end(), 0);
  }

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

template <typename T>
struct G2pPad {
  std::array<Vector3<T>, 27> positions{};
  std::array<Vector3<T>, 27> velocities{};

  void Reset() {
    std::fill(positions.begin(), positions.end(), Vector3<T>::Zero());
    std::fill(velocities.begin(), velocities.end(), Vector3<T>::Zero());
  }

  // @pre a, b, c ∈ {−1,0,1}
  void SetPositionAndVelocityAt(int a, int b, int c, const Vector3<T>& position,
                                const Vector3<T>& velocity) {
    int idx_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
    positions[idx_local] = position;
    velocities[idx_local] = velocity;
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
