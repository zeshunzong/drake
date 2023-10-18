#include "drake/multibody/mpm/sparse_grid.h"
namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
SparseGrid<T>::SparseGrid(double h)
    : num_active_nodes_(0), h_(h), active_nodes_(0) {}

template <typename T>
void SparseGrid<T>::Reserve(size_t capacity) {
  index_map_.reserve(capacity);
  active_nodes_.reserve(capacity);
  velocities_.reserve(capacity);
  masses_.reserve(capacity);
  forces_.reserve(capacity);
  batch_sizes_.reserve(capacity);
}

template <typename T>
void SparseGrid<T>::AppendActiveNode(const Vector3<int>& index_3d) {
  if (index_map_.count(index_3d) == 0) {
    // if this node has not been marked active, mark it as active
    active_nodes_.push_back(index_3d);
    ++num_active_nodes_;
    index_map_[index_3d] = 1;  // temporary value, will be updated in sorting
  }
}

template <typename T>
void SparseGrid<T>::ClearActiveNodes() {
  num_active_nodes_ = 0;
  active_nodes_.clear();
  index_map_.clear();
}

template <typename T>
void SparseGrid<T>::SortActiveNodes() {
  DRAKE_DEMAND(num_active_nodes_ == active_nodes_.size());

  std::sort(active_nodes_.begin(), active_nodes_.end(),
            internal::CompareIndex3DLexicographically);

  for (size_t i = 0; i < num_active_nodes_; ++i) {
    index_map_[active_nodes_[i]] = i;
  }
}

template <typename T>
void SparseGrid<T>::UpdateActiveNodesFromParticlePositions(
    const std::vector<Vector3<T>>& positions) {
  ClearActiveNodes();  // clear exisiting active nodes
  // first get the batch index for each particle
  const std::vector<Vector3<int>> particles_batch_indices =
      ComputeBatchIndices(positions);
  for (size_t i = 0; i < particles_batch_indices.size(); ++i) {
    // loop over each particle's batch index

    const Vector3<int>& center_node = particles_batch_indices[i];
    // center_node is the center of all 27 nodes neighboring to positions[i]
    Vector3<int> neighbor_node;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // get all 27 neighboring grid nodes to particle p
          neighbor_node = {center_node(0) + a, center_node(1) + b,
                           center_node(2) + c};

          if (index_map_.count(neighbor_node) == 0) {
            // if this node is not active, mark as active here
            active_nodes_.emplace_back(std::move(neighbor_node));
            num_active_nodes_++;
            index_map_[neighbor_node] = 1;
            // temporarily set it to 1, will sort it to get its 1D index
          }
          // should be equivalent to AppendActiveNode(neighbor_node)
          // emplace_back?
        }
      }
    }
  }
  SortActiveNodes();
  ResizeMassesForcesVelocities();
}

template <typename T>
void SparseGrid<T>::ResizeMassesForcesVelocities() {
  DRAKE_DEMAND(num_active_nodes_ == active_nodes_.size());
  masses_.resize(num_active_nodes_);
  velocities_.resize(num_active_nodes_);
  forces_.resize(num_active_nodes_);
}

template <typename T>
internal::TotalMassMomentum<T> SparseGrid<T>::ComputeTotalMassMomentum() const {
  DRAKE_DEMAND(masses_.size() == num_active_nodes_);
  DRAKE_DEMAND(velocities_.size() == num_active_nodes_);
  internal::TotalMassMomentum<T> total_mass_momentum;
  total_mass_momentum.total_mass = 0.0;
  total_mass_momentum.total_momentum = Vector3<T>::Zero();
  total_mass_momentum.total_angular_momentum = Vector3<T>::Zero();
  Vector3<T> position;
  for (size_t i = 0; i < num_active_nodes_; ++i) {
    position = GetPositionAt(Expand1DIndex(i));
    total_mass_momentum.total_mass += masses_[i];
    total_mass_momentum.total_momentum += masses_[i] * velocities_[i];
    total_mass_momentum.total_angular_momentum +=
        masses_[i] * position.cross(velocities_[i]);
  }
  return total_mass_momentum;
}

template <typename T>
void SparseGrid<T>::ResetMassesForcesVelocities() {
  DRAKE_ASSERT(masses_.size() == num_active_nodes_);
  DRAKE_ASSERT(forces_.size() == num_active_nodes_);
  DRAKE_ASSERT(velocities_.size() == num_active_nodes_);
  std::fill(masses_.begin(), masses_.begin() + num_active_nodes_, 0.0);
  std::fill(forces_.begin(), forces_.begin() + num_active_nodes_,
            Vector3<T>::Zero());
  std::fill(velocities_.begin(), velocities_.begin() + num_active_nodes_,
            Vector3<T>::Zero());
}

template <typename T>
std::vector<std::vector<int>> SparseGrid<T>::CalcGridHessianSparsityPattern()
    const {
  // the value to be returned
  std::vector<std::vector<int>> pattern;
  Vector3<int> current_index_3d;
  // loop over all active grid nodes
  for (size_t index1d = 0; index1d < num_active_nodes_; ++index1d) {
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
            if (Reduce3DIndex(neighbor_node_index_3d) >= index1d) {
              // we also require that this block should be in the upper right
              pattern_i.push_back(
                  static_cast<int>(Reduce3DIndex(neighbor_node_index_3d)));
            }
          }
        }
      }
    }
    pattern.push_back(std::move(pattern_i));
  }
  return pattern;
}

template <typename T>
std::vector<Vector3<int>> SparseGrid<T>::ComputeBatchIndices(
    const std::vector<Vector3<T>>& positions) const {
  using std::round;
  std::vector<Vector3<int>> batch_indices;

  for (size_t i = 0; i < positions.size(); ++i) {
    batch_indices.emplace_back(
        Vector3<int>{static_cast<int>(round(positions[i](0) / h_)),
                     static_cast<int>(round(positions[i](1) / h_)),
                     static_cast<int>(round(positions[i](2) / h_))});
  }
  return batch_indices;
}

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
