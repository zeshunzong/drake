#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void SparseGrid<T>::ComputeBatchIndices(
    const std::vector<Vector3<T>>& positions) {
  using std::round;
  batch_indices_.resize(positions.size());

  for (size_t i = 0; i < positions.size(); ++i) {
    batch_indices_[i] =
        Vector3<int>{static_cast<int>(round(positions[i](0) / h_)),
                     static_cast<int>(round(positions[i](1) / h_)),
                     static_cast<int>(round(positions[i](2) / h_))};
  }
}

template <typename T>
void SparseGrid<T>::Update(const std::vector<Vector3<T>>& positions) {
  ClearActiveNodes();  // clear exisiting active nodes

  // first compute the batch index for each particle
  ComputeBatchIndices(positions);

  for (size_t i = 0; i < batch_indices_.size(); ++i) {
    // loop over each particle's batch index

    const Vector3<int>& center_node = batch_indices_[i];
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
            map_from_1d_to_3d_.emplace_back(std::move(neighbor_node));
            map_from_3d_to_1d_[neighbor_node] = SIZE_MAX;
          }
        }
      }
    }
  }
  SortActiveNodes();
  ResetBatchSizes();
  // then calculate the number of particles in each batch
  for (size_t i = 0; i < batch_indices_.size(); ++i) {
    // loop over each particle (and its batch_index)
    // now the two bijective maps have been created
    ++batch_sizes_[Reduce3DIndex(batch_indices_[i])];
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

template <typename T>
std::vector<std::vector<int>> SparseGrid<T>::CalcGridHessianSparsityPattern()
    const {
  // the value to be returned
  std::vector<std::vector<int>> pattern;
  Vector3<int> current_index_3d;
  size_t neighbornode_index_1d; 
  // loop over all active grid nodes
  for (size_t index1d = 0; index1d < num_active_nodes(); ++index1d) {
    current_index_3d = Expand1DIndex(index1d);
    std::vector<int> pattern_i;
    // get its 125 neighbors
    for (int ic = -2; ic <= 2; ++ic) {
      for (int ib = -2; ib <= 2; ++ib) {
        for (int ia = -2; ia <= 2; ++ia) {
          Vector3<int> neighbor_node_index_3d =
              current_index_3d +
              Vector3<int>(ia, ib, ic);  // global 3d index of its neighbor
          if (IsActive(neighbor_node_index_3d)) {
            // we first require this is an active grid node
            neighbornode_index_1d = Reduce3DIndex(neighbor_node_index_3d);
            if (neighbornode_index_1d >= index1d) {
              // we also require that this block should be in the upper right
              pattern_i.push_back(
                  static_cast<int>(neighbornode_index_1d));
            }
          }
        }
      }
    }
    pattern.push_back(std::move(pattern_i));
  }
  return pattern;
}

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
