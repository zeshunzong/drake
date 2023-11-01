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
  void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles);

  /**
   * Particles to grid transfer.
   * See Section 10.1 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   */
  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* grid_data);

  /**
   * Grid to particles transfer
   * more tbd
   */
  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data,
           Particles<T>* particles, double dt) {
    DRAKE_ASSERT(dt > 0.0);
    size_t p_start, p_end, idx_local;
    Pad<T> local_pad;
    Vector3<int> batch_index_3d;
    // For each batch of particles
    p_start = 0;
    for (size_t i = 0; i < grid.num_active_nodes(); ++i) {
      p_end = p_start + grid.GetBatchSizeAtNode(i);
      if (p_start == p_end) {
        continue;
      }  // Skip empty batches
      batch_index_3d = grid.Expand1DIndex(i);

      for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
          for (int c = -1; c <= 1; ++c) {
            idx_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
            Vector3<int> index_3d =
                Vector3<int>(batch_index_3d(0) + a, batch_index_3d(1) + b,
                             batch_index_3d(2) + c);
            // batch_states[idx_local].position = grid.get_position(index_3d);
            local_pad.pad_velocities[idx_local] =
                grid_data.velocities_[grid.Reduce3DIndex(index_3d)];
          }
        }
      }
      // For each particle in the batch, update the particles' states
      for (size_t p = p_start; p < p_end; ++p) {
        particles->UpdateVelocityParticleFromItsPad(p, local_pad);
      }
      p_start = p_end;
    }
    particles->AdvectParticles(dt);
  }

 private:
  // scratch pads for transferring states from particles to grid nodes
  // local_pads_.size() = SparseGrid.num_active_nodes()
  std::vector<Pad<T>> local_pads_{};

  // Add the temporary data stored in bath_index-th pad into grid_data
  void AddPadToGridData(const Vector3<int>& batch_index, const Pad<T>& pad,
                        const SparseGrid<T>& grid,
                        GridData<T>* grid_data) const;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
