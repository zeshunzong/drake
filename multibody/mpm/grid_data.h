#pragma once

#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A container for the kinematic data that will be stored on grid nodes.
 * In particular, they are
 *      mass on each active node
 *      momentum on each active node
 *      velocity on each active node
 *      force on each active node
 */
template <typename T>
class GridData {
 public:
  std::vector<T> masses_;
  std::vector<Vector3<T>> momentums_;
  std::vector<Vector3<T>> velocities_;
  std::vector<Vector3<T>> forces_;

  void Reserve(size_t capacity) {
    masses_.reserve(capacity);
    momentums_.reserve(capacity);
    velocities_.reserve(capacity);
    forces_.reserve(capacity);
  }

  /**
   * Resets the containers to the correct size.
   * Sets all values to zero.
   */
  void Reset(size_t num_active_nodes) {
    masses_.resize(num_active_nodes);
    momentums_.resize(num_active_nodes);
    velocities_.resize(num_active_nodes);
    forces_.resize(num_active_nodes);
    std::fill(masses_.begin(), masses_.end(), 0.0);
    std::fill(momentums_.begin(), momentums_.end(), Vector3<T>::Zero());
    std::fill(forces_.begin(), forces_.end(), Vector3<T>::Zero());
    std::fill(velocities_.begin(), velocities_.end(), Vector3<T>::Zero());
  }

  /**
   * Adds mass, momentum, and force to the node at index_1d.
   */
  void AccumulateAt(size_t index_1d, const T& mass, const Vector3<T>& momentum,
                    const Vector3<T>& force) {
    DRAKE_ASSERT(index_1d < masses_.size());
    masses_[index_1d] += mass;
    momentums_[index_1d] += momentum;
    forces_[index_1d] += force;
  }

  /**
   * Computes the velocity for momentum for each active grid node.
   * @pre masses_[i] > 0 for all i. This will be satisfied if only active grid
   * nodes are stored and grid data for each active grid node has already been
   * accumulated.
   */
  void ComputeVelocitiesFromMomentums() {
    for (size_t i = 0; i < masses_.size(); ++i) {
      velocities_[i] = momentums_[i] / masses_[i];
    }
  }

  /**
   * @pre index_1d < number of active nodes stored
   */
  const Vector3<T>& GetVelocityAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < velocities_.size());
    return velocities_[index_1d];
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
