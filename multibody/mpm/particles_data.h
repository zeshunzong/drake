#pragma once

#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * Intermediary data for updating Particles.
 * B-matrix, velocity, and grad_v will be computed in MpmTransfer.G2P() and
 * stored here.
 */
template <typename T>
struct ParticlesData {
  std::vector<Matrix3<T>> particle_B_matrices_next{};
  std::vector<Vector3<T>> particle_velocites_next{};
  std::vector<Matrix3<T>> particle_grad_v_next{};

  void Resize(size_t s) {
    particle_B_matrices_next.resize(s);
    particle_velocites_next.resize(s);
    particle_grad_v_next.resize(s);
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
