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

/**
 * Intermediary data for computing elastic energy and its derivatives.
 * Elastic deformation gradient is computed for each particle using grid
 * velocities, other quantities are derived from the elastic deformation
 * gradient.
 * @note elastic F is needed in computing elastic energy.
 * @note PK_stress is needed in computing grid elastic force.
 * @note dPdF is needed in computing the hessian of elastic energy w.r.t. grid
 * positions/velocities.
 */
template <typename T>
struct DeformationScratch {
  std::vector<Matrix3<T>> elastic_deformation_gradients{};
  std::vector<Matrix3<T>> PK_stresses{};
  std::vector<Eigen::Matrix<T, 9, 9>> dPdF{};

  void Resize(size_t s) {
    elastic_deformation_gradients.resize(s);
    PK_stresses.resize(s);
    dPdF.resize(s);
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
