#pragma once

#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/internal/mass_and_momentum.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A container for the kinematic data that will be stored on grid nodes.
 * In particular, they are
 *      mass on each node
 *      momentum / velocity on each node (they differ by a scalar = node mass)
 *      force on each node
 */
template <typename T>
struct GridData {
  std::vector<T> masses;
  std::vector<Vector3<T>> velocities;
  std::vector<Vector3<T>> forces;

  void Reserve(size_t capacity) {
    masses.reserve(capacity);
    velocities.reserve(capacity);
    forces.reserve(capacity);
  }

  /**
   * Resets the container to the correct size.
   * Resets all values to zero.
   */
  void Reset(size_t num_active_nodes) {
    masses.resize(num_active_nodes);
    velocities.resize(num_active_nodes);
    forces.resize(num_active_nodes);
    std::fill(masses.begin(), masses.begin() + num_active_nodes, 0.0);
    std::fill(forces.begin(), forces.begin() + num_active_nodes,
              Vector3<T>::Zero());
    std::fill(velocities.begin(), velocities.begin() + num_active_nodes,
              Vector3<T>::Zero());
  }

  /**
   * If momentum is stored, converts it to velocity.
   * @note By construction active grid nodes will always have positive mass, so
   * the division is safe.
   */
  void ConvertMomentumToVelocity() {
    for (size_t i = 0; i < velocities.size(); ++i) {
      velocities[i] = velocities[i] / masses[i];
    }
  }

  /**
   * Returns the total mass and momentum, from the grid's perspective.
   */
  internal::MassAndMomentum<T> ComputeTotalMassAndMomentumGrid(
      const SparseGrid<T>& grid) const {
    internal::MassAndMomentum<T> total_mass_momentum{};
    for (size_t i = 0; i < masses.size(); ++i) {
      total_mass_momentum.total_mass += masses[i];
      total_mass_momentum.total_momentum += masses[i] * velocities[i];
      total_mass_momentum.total_angular_momentum +=
          masses[i] *
          grid.GetPositionAt(grid.Expand1DIndex(i)).cross(velocities[i]);
    }

    return total_mass_momentum;
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
