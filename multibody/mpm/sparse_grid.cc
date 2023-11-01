#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void SparseGrid<T>::ComputeBatchIndices(
    const std::vector<Vector3<T>>& positions,
    std::vector<Vector3<int>>* batch_indices) {
  using std::round;
  DRAKE_DEMAND(batch_indices->size() == positions.size());

  for (size_t i = 0; i < positions.size(); ++i) {
    (*batch_indices)[i] =
        Vector3<int>{static_cast<int>(round(positions[i](0) / h_)),
                     static_cast<int>(round(positions[i](1) / h_)),
                     static_cast<int>(round(positions[i](2) / h_))};
  }
}

template <typename T>
void SparseGrid<T>::Update(const std::vector<Vector3<T>>& positions,
                           std::vector<Vector3<int>>* batch_indices) {
  ClearActiveNodes();  // clear exisiting active nodes

  // first compute the batch index for each particle
  ComputeBatchIndices(positions, batch_indices);

  for (size_t i = 0; i < batch_indices->size(); ++i) {
    // loop over each particle's batch index

    const Vector3<int>& center_node = (*batch_indices)[i];
    // center_node is the center of all 27 nodes neighboring to positions[i]
    Vector3<int> neighbor_node;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // get all 27 neighboring grid nodes to particle p
          neighbor_node = {center_node(0) + a, center_node(1) + b,
                           center_node(2) + c};

          if (map_from_3d_to_1d_.count(neighbor_node) == 0) {
            // if this node is not active, mark as active here
            map_from_1d_to_3d_.emplace_back(neighbor_node);
            map_from_3d_to_1d_[neighbor_node] =
                std::numeric_limits<size_t>::max();
          }
        }
      }
    }
  }
  SortActiveNodes();
  ResetBatchSizes();
  // then calculate the number of particles in each batch
  for (size_t i = 0; i < batch_indices->size(); ++i) {
    // loop over each particle (and its batch_index)
    // now the two bijective maps have been created
    ++batch_sizes_[Reduce3DIndex((*batch_indices)[i])];
  }
}

template <typename T>
void SparseGrid<T>::ClearActiveNodes() {
  map_from_3d_to_1d_.clear();
  map_from_1d_to_3d_.clear();
}

template <typename T>
void SparseGrid<T>::SortActiveNodes() {
  std::sort(map_from_1d_to_3d_.begin(), map_from_1d_to_3d_.end(),
            internal::CompareIndex3DLexicographically);

  for (size_t i = 0; i < num_active_nodes(); ++i) {
    map_from_3d_to_1d_[map_from_1d_to_3d_[i]] = i;
  }
}

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
