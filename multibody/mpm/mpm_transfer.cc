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

template class MpmTransfer<double>;
template class MpmTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
