#pragma once

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/internal/hashing_utils.h"
#include "drake/multibody/mpm/internal/math_utils.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class SparseGrid {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SparseGrid);

  explicit SparseGrid(double h) : h_(h), Dp_inv_(4.0 / h / h) {}

  void Reserve(size_t capacity) {
    map_from_1d_to_3d_.reserve(capacity);
    map_from_3d_to_1d_.reserve(capacity);
    batch_sizes_.reserve(capacity);
  }

  // Updates the grid active nodes, and sort them.
  void Update(const std::vector<Vector3<T>>& positions);

  size_t num_active_nodes() const { return map_from_1d_to_3d_.size(); }

  double h() const { return h_; }

  const std::vector<size_t>& batch_sizes() { return batch_sizes_; }

  const std::vector<Vector3<int>>& batch_indices() { return batch_indices_; }

  /**
   * Checks whether the grid node at position (ih, jh, kh) is active or not,
   * where we denote {i, j, k} = index_3d.
   */
  bool IsActive(const Vector3<int>& index_3d) const {
    return map_from_3d_to_1d_.count(index_3d) == 1;
  }

  /**
   * Reduces an 3D index_3d = {i, j, k} index to its corresponding 1D index on
   * the sparse grid.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  size_t Reduce3DIndex(const Vector3<int>& index_3d) const {
    return map_from_3d_to_1d_.at(index_3d);
  }

  /**
   * Expands a lexicographical index_1d to its corresponding 3D index (i,j,k).
   * @pre index_1d < num_active_nodes()
   */
  const Vector3<int>& Expand1DIndex(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes());
    return map_from_1d_to_3d_[index_1d];
  }

 private:
  void ClearActiveNodes();

  void SortActiveNodes();

  /**
   * For each position x in positions, computes the 3d grid index {i, j, k} such
   * that (ih, jh, kh) is the closet grid node to position x. When the position
   * x is the position of a particle, the corresponding 3d grid index {i, j, k}
   * is called the batch index of this particle.
   * @note the returned std::vector has the same size as positions
   */
  void ComputeBatchIndices(const std::vector<Vector3<T>>& positions);

  void ResetBatchSizes() {
    batch_sizes_.resize(num_active_nodes());
    std::fill(batch_sizes_.begin(), batch_sizes_.end(), 0);
  }

  double h_{};
  double Dp_inv_{};
  std::vector<Vector3<int>> map_from_1d_to_3d_;
  std::unordered_map<Vector3<int>, size_t> map_from_3d_to_1d_;
  std::vector<size_t> batch_sizes_{};

  std::vector<Vector3<int>> batch_indices_{};  // size = num of particles, do we
                                               // reserve? do we put it here?
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
