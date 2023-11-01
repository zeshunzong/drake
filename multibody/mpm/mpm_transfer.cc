#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MpmTransfer<T>::SetUpTransfer(SparseGrid<T>* grid,
                                   Particles<T>* particles) {
  if (grid->num_active_nodes() == 0) {
    // this is indeed an upper bound
    // this is called only once through the entire simulation
    grid->Reserve(27 * particles->num_particles());
    local_pads_.reserve(27 * particles->num_particles());
  }
  if (particles->grid_and_particles_and_splatters_need_update()) {
    // identify active grid nodes and sort them lexicographically
    grid->Update(particles->positions(),
                 &(particles->GetMutableBatchIndices()));
    // sort particles, update weights and dweights
    particles->SortParticlesAndUpdateSplatters(*grid);
  }
}

template <typename T>
void MpmTransfer<T>::P2G(const Particles<T>& particles,
                         const SparseGrid<T>& grid, GridData<T>* grid_data) {
  DRAKE_DEMAND(!particles.grid_and_particles_and_splatters_need_update());
  size_t p_start, p_end;
  // create num_active_nodes() local pads
  local_pads_.clear();
  local_pads_.resize(grid.num_active_nodes());
  grid_data->Reset(grid.num_active_nodes());
  p_start = 0;
  for (size_t batch_i = 0; batch_i < grid.num_active_nodes(); ++batch_i) {
    p_end = p_start + grid.GetBatchSizeAtNode(batch_i);
    for (size_t p = p_start; p < p_end; ++p) {
      particles.SplatOneParticleToItsPad(p, grid, &(local_pads_[batch_i]));
    }
    p_start = p_end;
  }
  for (size_t i = 0; i < grid.num_active_nodes(); ++i) {
    if (grid.GetBatchSizeAtNode(i) != 0) {
      // Put sums of local scratch pads to grid
      AddPadToGridData(grid.Expand1DIndex(i), local_pads_[i], grid, grid_data);
    }
  }
  grid_data->ConvertMomentumToVelocity();
}

template <typename T>
void MpmTransfer<T>::AddPadToGridData(const Vector3<int>& batch_index,
                                      const Pad<T>& pad,
                                      const SparseGrid<T>& grid,
                                      GridData<T>* grid_data) const {
  int node_index_local;
  Vector3<int> node_index_3d;

  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      for (int c = -1; c <= 1; ++c) {
        node_index_3d = batch_index + Vector3<int>{a, b, c};
        node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);

        grid_data->masses_[grid.Reduce3DIndex(node_index_3d)] +=
            pad.pad_masses[node_index_local];
        grid_data->velocities_[grid.Reduce3DIndex(node_index_3d)] +=
            pad.pad_velocities[node_index_local];
        grid_data->forces_[grid.Reduce3DIndex(node_index_3d)] +=
            pad.pad_forces[node_index_local];
      }
    }
  }
}

template class MpmTransfer<double>;
template class MpmTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
