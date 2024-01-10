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
                         const SparseGrid<T>& grid, GridData<T>* grid_data,
                         TransferScratch<T>* scratch) const {
  particles.SplatToP2gPads(grid.h(), &(scratch->p2g_pads));
  grid.GatherFromP2gPads(scratch->p2g_pads, grid_data);
}

template <typename T>
void MpmTransfer<T>::G2P(const SparseGrid<T>& grid,
                         const GridData<T>& grid_data,
                         const Particles<T>& particles,
                         ParticlesData<T>* particles_data,
                         TransferScratch<T>* scratch) const {
  DRAKE_DEMAND(!particles.NeedReordering());
  particles_data->Resize(particles.num_particles());

  Vector3<int> idx_3d;
  // loop over all batches
  Vector3<int> batch_idx_3d;
  const std::vector<Vector3<int>>& base_nodes = particles.base_nodes();
  const std::vector<size_t>& batch_starts = particles.batch_starts();
  for (size_t i = 0; i < particles.num_batches(); ++i) {
    batch_idx_3d = base_nodes[batch_starts[i]];

    // form the g2p_pad for this batch
    scratch->g2p_pad.SetZero();
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_3d = Vector3<int>(batch_idx_3d(0) + a, batch_idx_3d(1) + b,
                                batch_idx_3d(2) + c);
          const Vector3<double> position =
              internal::ComputePositionFromIndex3D(idx_3d, grid.h());
          const Vector3<T>& velocity =
              grid_data.GetVelocityAt(grid.To1DIndex(idx_3d));

          scratch->g2p_pad.SetPositionAndVelocityAt(a, b, c, position,
                                                    velocity);
        }
      }
    }
    // write particle v, B, and grad v to particles_data
    particles.WriteParticlesDataFromG2pPad(i, scratch->g2p_pad, particles_data);
  }
}

template <typename T>
void MpmTransfer<T>::UpdateParticlesState(
    const ParticlesData<T>& particles_data, double dt,
    Particles<T>* particles) const {
  particles->SetVelocities(particles_data.particle_velocites_next);
  particles->SetBMatrices(particles_data.particle_B_matrices_next);
  particles->UpdateTrialDeformationGradients(
      dt, particles_data.particle_grad_v_next);
  // TODO(zeshunzong): compute new projected F and new stress
}

template class MpmTransfer<double>;
template class MpmTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
