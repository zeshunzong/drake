#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MpmTransfer<T>::SetUpTransfer(SparseGrid<T>* grid,
                                   Particles<T>* particles) const {
  particles->Prepare(grid->h());
  grid->MarkActiveNodes(particles->base_nodes());
}

template <typename T>
void MpmTransfer<T>::P2G(const Particles<T>& particles,
                         const SparseGrid<T>& grid, GridData<T>* grid_data) {
  particles.SplatToPads(grid.h(), &pads_);
  grid.GatherFromPads(pads_, grid_data);
}

template <typename T>
void MpmTransfer<T>::G2P(const SparseGrid<T>& grid,
                         const GridData<T>& grid_data, double dt,
                         Particles<T>* particles) {
  DRAKE_DEMAND(dt > 0.0);
  DRAKE_DEMAND(!particles->NeedReordering());

  Vector3<int> idx_3d;

  // loop over all batches
  Vector3<int> batch_idx_3d;
  const std::vector<Vector3<int>>& base_nodes = particles->base_nodes();
  const std::vector<size_t>& batch_starts = particles->batch_starts();
  for (size_t i = 0; i < particles->num_batches(); ++i) {
    batch_idx_3d = base_nodes[batch_starts[i]];

    // form the batch_pad for this batch
    batch_pad_.Reset();
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_3d = Vector3<int>(batch_idx_3d(0) + a, batch_idx_3d(1) + b,
                                batch_idx_3d(2) + c);
          const Vector3<double> position =
              internal::ComputePositionFromIndex3D(idx_3d, grid.h());
          const Vector3<T>& velocity =
              grid_data.GetVelocityAt(grid.To1DIndex(idx_3d));

          batch_pad_.SetPositionAndVelocityAt(a, b, c, position, velocity);
        }
      }
    }

    // update particles in this batch, excluding positions
    particles->UpdateBatchParticlesFromBatchPad(i, dt, batch_pad_);
  }

  particles->AdvectParticles(dt);
}

template class MpmTransfer<double>;
template class MpmTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
