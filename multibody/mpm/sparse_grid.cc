#include "drake/multibody/mpm/sparse_grid.h"
namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
SparseGrid<T>::SparseGrid(double h) : h_(h) {
  DRAKE_DEMAND(h > 0.0);
}

template <typename T>
void SparseGrid<T>::Reserve(size_t capacity) {
  map_1d_to_3d_.reserve(capacity);
  map_3d_to_1d_.reserve(capacity);
}

template <typename T>
void SparseGrid<T>::MarkActiveNodes(
    const std::vector<Vector3<int>>& base_nodes) {
  map_1d_to_3d_.clear();
  map_3d_to_1d_.clear();
  size_t count = 0;
  for (size_t i = 0; i < base_nodes.size(); ++i) {
    const Vector3<int>& base_node = base_nodes[i];
    Vector3<int> neighbor_node;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // get all 27 neighboring grid nodes to this base node
          neighbor_node = {base_node(0) + a, base_node(1) + b,
                           base_node(2) + c};

          if (map_3d_to_1d_.count(neighbor_node) == 0) {
            // if this node is not active, mark as active here
            map_1d_to_3d_.emplace_back(neighbor_node);
            map_3d_to_1d_[neighbor_node] = count;
            ++count;
          }
        }
      }
    }
  }
}

template <typename T>
void SparseGrid<T>::GatherFromP2gPads(const std::vector<P2gPad<T>>& p2g_pads,
                                      GridData<T>* grid_data) const {
  grid_data->Reset(num_active_nodes());
  for (const P2gPad<T>& p2g_pad : p2g_pads) {
    const Vector3<int>& base_node = p2g_pad.base_node;
    // Add pad data to grid data.
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          size_t index_1d = To1DIndex(Vector3<int>(a, b, c) + base_node);
          grid_data->AccumulateAt(index_1d, p2g_pad.GetMassAt(a, b, c),
                                  p2g_pad.GetMomentumAt(a, b, c),
                                  p2g_pad.GetForceAt(a, b, c));
        }
      }
    }
  }
  grid_data->ComputeVelocitiesFromMomentums();
}

template <typename T>
void SparseGrid<T>::GatherForceFromP2gPads(
    const std::vector<P2gPad<T>>& p2g_pads,
    std::vector<Vector3<T>>* grid_forces) const {
  std::vector<Vector3<T>>& grid_forces_ref = *grid_forces;
  grid_forces_ref.resize(num_active_nodes(), Vector3<T>::Zero());

  for (const P2gPad<T>& p2g_pad : p2g_pads) {
    const Vector3<int>& base_node = p2g_pad.base_node;
    // Add pad data to grid data.
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          size_t index_1d = To1DIndex(Vector3<int>(a, b, c) + base_node);
          grid_forces_ref[index_1d] += p2g_pad.GetForceAt(a, b, c);
        }
      }
    }
  }
}

template <typename T>
internal::MassAndMomentum<T> SparseGrid<T>::ComputeTotalMassMomentum(
    const GridData<T>& grid_data) const {
  internal::MassAndMomentum<T> total_mass_momentum{};
  for (size_t i = 0; i < num_active_nodes(); ++i) {
    total_mass_momentum.total_mass += grid_data.masses_[i];
    total_mass_momentum.total_momentum +=
        grid_data.masses_[i] * grid_data.velocities_[i];
    const Vector3<T> node_position =
        internal::ComputePositionFromIndex3D(To3DIndex(i), h_);
    total_mass_momentum.total_angular_momentum +=
        grid_data.masses_[i] * node_position.cross(grid_data.velocities_[i]);
  }
  return total_mass_momentum;
}

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
