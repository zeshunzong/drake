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
 * transfer.SetUpTransfer(grid, particles);
 * transfer.P2G(particles, grid, &grid_data); // P2G
 *
 * // grid update tbd
 *
 * // g2p tbd
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

 private:
  // scratch pads for transferring states from particles to grid nodes
  std::vector<Pad<T>> pads_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
