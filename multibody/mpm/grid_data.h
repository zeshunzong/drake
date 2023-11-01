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
class GridData {
 public:
  std::vector<T> masses_;
  std::vector<Vector3<T>> velocities_;
  std::vector<Vector3<T>> forces_;

  void Reserve(size_t capacity) {
    masses_.reserve(capacity);
    velocities_.reserve(capacity);
    forces_.reserve(capacity);
  }

  /**
   * Resets the container to the correct size.
   * Resets all values to zero.
   */
  void Reset(size_t num_active_nodes) {
    masses_.resize(num_active_nodes);
    velocities_.resize(num_active_nodes);
    forces_.resize(num_active_nodes);
    std::fill(masses_.begin(), masses_.end(), 0.0);
    std::fill(forces_.begin(), forces_.end(), Vector3<T>::Zero());
    std::fill(velocities_.begin(), velocities_.end(), Vector3<T>::Zero());
  }

  /**
   * If momentum is stored, converts it to velocity.
   * @note By construction active grid nodes will always have positive mass, so
   * the division is safe.
   */
  void ConvertMomentumToVelocity() {
    for (size_t i = 0; i < velocities_.size(); ++i) {
      velocities_[i] = velocities_[i] / masses_[i];
    }
  }

  /**
   * Returns the total mass and momentum, from the grid's perspective.
   */
  internal::MassAndMomentum<T> ComputeTotalMassAndMomentumGrid(
      const SparseGrid<T>& grid) const {
    internal::MassAndMomentum<T> total_mass_momentum{};
    for (size_t i = 0; i < masses_.size(); ++i) {
      total_mass_momentum.total_mass += masses_[i];
      total_mass_momentum.total_momentum += masses_[i] * velocities_[i];
      total_mass_momentum.total_angular_momentum +=
          masses_[i] *
          grid.GetPositionAt(grid.Expand1DIndex(i)).cross(velocities_[i]);
    }

    return total_mass_momentum;
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
