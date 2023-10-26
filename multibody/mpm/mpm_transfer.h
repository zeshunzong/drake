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
  // DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PadSplatter);

  MpmTransfer() {}

  /**
   * Given current configuration of grid and particles, performs all necessary
   * preparations for transferring, including sorting and computing weights.
   *
   */
  void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles) const;

  /**
   * Particles to grid transfer.
   * See Section 10.1 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   */
  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* grid_data) const;

 private:
  /**
   * Add the temporary data stored in bath_index-th pad into grid_data
   */
  void AddPadToGridData(const Vector3<int>& batch_index, const Pad<T>& pad,
                        const SparseGrid<T>& grid,
                        GridData<T>* grid_data) const;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
