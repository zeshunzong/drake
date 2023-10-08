#include "drake/multibody/mpm/particles.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
Particles<T>::Particles() : num_particles_(0) {}

template <typename T>
Particles<T>::Particles(size_t num_particles)
    : num_particles_(num_particles),
      positions_(num_particles),
      velocities_(num_particles),
      masses_(num_particles),
      reference_volumes_(num_particles),
      deformation_gradients_(num_particles),
      B_matrices_(num_particles),
      neighbor_grid_nodes_global_indices_(num_particles),
      w_ip_neighbor_nodes_(num_particles),
      dw_ip_neighbor_nodes_(num_particles) {}

template <typename T>
void Particles<T>::Reorder(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles_);
  for (size_t i = 0; i < num_particles_ - 1; ++i) {
    // don't need to sort the last element if the first (n-1) elements have
    // already been sorted
    size_t ind = new_order[i];
    // the i-th element should be placed at ind-th position
    if (ind < i) {
      // its correct position is before i. In this case, the element must have
      // already been swapped to a position after i. find out where it has been
      // swapped to.
      while (ind < i) {
        ind = new_order[ind];
      }
    }
    // after this operation, ind is either equal to i or larger than i
    if (ind == i) {
      // at its correct position, nothing needs to be done
      continue;
    } else if (ind > i) {
      // TODO(zeshunzong): update this as more attributes are added

      std::swap(masses_[i], masses_[ind]);
      std::swap(reference_volumes_[i], reference_volumes_[ind]);

      std::swap(positions_[i], positions_[ind]);
      std::swap(velocities_[i], velocities_[ind]);

      std::swap(deformation_gradients_[i], deformation_gradients_[ind]);
      std::swap(B_matrices_[i], B_matrices_[ind]);
    } else {
      DRAKE_UNREACHABLE();
    }
  }
}

template <typename T>
void Particles<T>::AddParticle(const Vector3<T>& position,
                               const Vector3<T>& velocity, T mass,
                               T reference_volume,
                               const Matrix3<T>& deformation_gradient,
                               const Matrix3<T>& B_matrix) {
  positions_.emplace_back(position);
  velocities_.emplace_back(velocity);
  masses_.emplace_back(mass);
  reference_volumes_.emplace_back(reference_volume);
  deformation_gradients_.emplace_back(deformation_gradient);
  B_matrices_.emplace_back(B_matrix);
  neighbor_grid_nodes_global_indices_.emplace_back();
  w_ip_neighbor_nodes_.emplace_back();
  dw_ip_neighbor_nodes_.emplace_back();
  num_particles_++;
}

template class Particles<double>;
template class Particles<AutoDiffXd>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
