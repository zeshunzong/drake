#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MPMTransfer<T>::SetUpTransfer(SparseGrid<T>* grid,
                                   Particles<T>* particles) const {
  if (grid->num_active_nodes() == 0) {
    // TODO(yiminlin.tri): Magic number
    // reserve memory for grid mass, velocity, and force
    // should only be called once throughout the entire simulation
    grid->Reserve(3 * particles->num_particles());
  }

  // identify active grid nodes and sort them lexicographically
  grid->Update(particles->positions());

  SortParticles(grid->batch_indices(), particles);

  // TODO: to be implemented
  particles->UpdatePadSplatters();// compute w, dw

}

template <typename T>
void MPMTransfer<T>::SortParticles(const std::vector<Vector3<int>> batch_indices,
                                   Particles<T>* particles) const {

  // Stores the particle index permutation after sorting
  std::vector<size_t> sorted_indices(particles->num_particles());

  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [grid, &batch_indices](size_t i1, size_t i2) {
              return internal::CompareIndex3DLexicographically(
                  batch_indices[i1], batch_indices[i2]);
            });

  // Reorder the particles
  particles->Reorder(sorted_indices);
}


template class MPMTransfer<double>;
template class MPMTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
