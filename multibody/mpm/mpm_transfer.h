#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/pad.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * An implementation of MPM's transfer schemes. We follow Section 10.5 in
 * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
 *
 * Temporal advancement from tₙ to tₙ₊₁ essentially consists of three steps.
 * 1) P2G
 * 2) Grid update (temporal update, implicit or explicit)
 * 3) G2P
 *
 * To do so, one needs the following objects:
 * Particles<double> particles;
 * SparseGrid<double> sparse_grid;
 * GridData<double> grid_data;
 * MpmTransfer<double> transfer;
 *
 * The above three steps are executed as
 *
 * // must setup the transfer before p2g, or whenever particle positions change
 * transfer.SetUpTransfer(&grid, &particles);
 * transfer.P2G(particles, grid, &grid_data); // P2G
 *
 * // grid update tbd
 *
 * transfer.G2P(grid, grid_data, dt, &particles); // G2P
 *
 * <-- TODO(zeshunzong): this is not complete and not exactly accurate for
 * implicit -->
 */
template <typename T>
class MpmTransfer {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmTransfer);

  MpmTransfer() {}

  /**
   * Given current configuration of grid and particles, performs all necessary
   * preparations for transferring, including sorting and computing weights.
   */
  void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles) const;

  /**
   * Particles to grid transfer.
   * See Section 10.1 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given particle data, writes to grid_data.
   * @note grid_data is first cleared before "transferring" particle data to
   * grid_data.
   */
  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* grid_data);

  /**
   * Grid to particles transfer.
   * See Section 10.1 and 10.2 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given grid_data, writes to particles.
   */
  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data, double dt,
           Particles<T>* particles);

  void G2P_another_option(const SparseGrid<T>& grid,
                          const GridData<T>& grid_data,
                          const Particles<T>& particles) {
    DRAKE_DEMAND(!particles.NeedReordering());

    // make sure the scratch_ has compatible size
    scratch_.Resize(particles.num_particles());
    
    Vector3<int> idx_3d;

    // loop over all batches
    Vector3<int> batch_idx_3d;
    const std::vector<Vector3<int>>& base_nodes = particles.base_nodes();
    const std::vector<size_t>& batch_starts = particles.batch_starts();
    for (size_t i = 0; i < particles.num_batches(); ++i) {
      batch_idx_3d = base_nodes[batch_starts[i]];

      // form the g2p_pad for this batch
      g2p_pad_.Reset();
      for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
          for (int c = -1; c <= 1; ++c) {
            idx_3d = Vector3<int>(batch_idx_3d(0) + a, batch_idx_3d(1) + b,
                                  batch_idx_3d(2) + c);
            const Vector3<double> position =
                internal::ComputePositionFromIndex3D(idx_3d, grid.h());
            const Vector3<T>& velocity =
                grid_data.GetVelocityAt(grid.To1DIndex(idx_3d));

            g2p_pad_.SetPositionAndVelocityAt(a, b, c, position, velocity);
          }
        }
      }

      // write particle v, B, and grad v to scratch
      particles.WriteBatchTimeIntegrationScratchFromG2pPad(i, g2p_pad_,
                                                           &scratch_);
    }
  }

  void ParticleTimeIntegrationAndUpdate(double dt, Particles<T>* particles) {
    // psudo code
    // particles->velocities = scratch_.particle_velocites_next
    // particles->B_matrices = scratch_.particle_B_matrices_next
    particles->AdvectParticles(dt);
    particles->UpdateTrialDeformationGradients(dt,
                                               scratch_.particle_grad_v_next);
  }

 private:
  // scratch pads for transferring states from particles to grid nodes
  std::vector<P2gPad<T>> p2g_pads_{};

  // scratch pad for transferring states from grid nodes to particles
  G2pPad<T> g2p_pad_{};

  TimeIntegrationScratch<T> scratch_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
