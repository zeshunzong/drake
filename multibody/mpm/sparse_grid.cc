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

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
