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
/// Implements a (3D) sparse grid that serves as the background Eulerian grid in
/// the Material Point Method. In MPM, a Cartesian grid is used to update
/// particle states. Particles reside in the grid as (schematically in 2D)
///           o - o - o - o - o - o - o
///           |   |   |   |x  |   |   |
///           o - o - o - o - o - o - o
///           |   |x x|x  |xx |   |   |
///           o - o - o - o - o - o - o
///           |   |x  |   |   |   |   |
///           o - o - o - o - o - o - o
///           |   |   |   |   |   |   |
///           o - o - o - o - o - o - o
///               Fig 1. Dense grid.
///
/// where     <-h-> h > 0 is the spacing between grid nodes,
///           o denotes a grid node,
/// and       x denotes a particle residing in the grid.
/// In MPM, the interpolation function we use (see b_spline.h) has a finite
/// support. Thus, each particle x only interacts with its "neighbor" grid
/// nodes. Therefore, a union of all the "neighbor" grid nodes would suffice for
/// the MPM algorithm. They are denoted as the "active grid nodes" for a sparse
/// grid. Schematically, those active grid nodes are
///                       o - o
///                       |x  |
///               o - o - o - o
///               |x x|x  |xx |
///               o - o - o - o
///               |x  |
///               o - o
///
///               Fig 2. Sparse grid.
///
/// @note This is only a schematic illustration. The actual number of active
/// grid nodes also depends on the interpolation function we choose.
///
/// @note Currently only quadratic B-spline is supported, where the support is
/// [-1.5h, 1.5h].
///
/// Given grid node spacing h, the grid nodes for the dense grid (c.f. Fig 1)
/// are characterized by the set D = {(i, j, k) ⋅ h | i, j, k ∈ ℤ}.
/// Correspondingly, the grid nodes for a sparse grid forms a subset S ⊂ D such
/// that each o ∈ S is a neighbor node of at least one particle x.
///
/// Under quadratic B-spline, each particle will activate its 27 neighbor grid
/// nodes in 3D (9 in 2D). See the illustration below.
///
///                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
///                ⋅    __⋅__    ⋅
///                ⋅   ▏  ⋅  ▕   ⋅
///                o ⋅ ▏⋅ X ⋅▕ ⋅ o
///                ⋅   ▏p_⋅__▕   ⋅
///                ⋅      ⋅      ⋅
///                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
/// Fig 3. Neighbor grid nodes activated by a particle p.
///
/// For a grid node i, any particle p that falls in an equidistant bounding box
/// with width 2×1.5h centered at i will result in a non-zero interpolation
/// function and thus interact with p. We can safely discard the boundary case
/// (i.e. a particle falls exactly onto the boundary of the bounding box) as the
/// interpolation function is nil at 1.5h. Equivalently, for a particle p, an
/// equidistant bounding box with width 2×1.5h centered at p will cover all grid
/// nodes whom p will interact with. In 1D this includes 3 nodes; in 2D this
/// includes 9 nodes (c.f. Fig 3); and in 3D this includes 27 nodes. In
/// particular, for a particle p, we can always find its nearest grid node
/// (denoted X in Fig 3). Write the 3d index of node X as {i,j,k}. Then nodes
/// indexed by {i±1, j±1, k±1} will all interact with particle p.
///
/// @note Following the above observation, we define {i,j,k} to be the (3D)
/// *batch index* for particle p, where (ih, jh, kh) is the position of grid
/// node X, c.f. Fig 3.
///
/// To summarize, we implement the MPM in 3D with quadratic B spline
/// interpolation function, and thus one particle will activate its 27 neighbor
/// grids.
///
/// @note The active grid nodes depend on the positions of particles. When the
/// particles move to new positions, the sparse grid is updated as some existing
/// active nodes cease to be active, while some new neighbor nodes join.
///
/// The collection of active grid nodes will be sorted lexicographically based
/// on their positions. Hence, an active grid node at position (ih, jh, kh) can
/// be accessed by either Vector3<int>{i,j,k} (denoted index_1d) or its
/// corresponding lexicographical order (denoted index_1d) among all active
/// nodes. index_1d ∈ {0, 1, ..., num_active_nodes()-1}.
///
/// @note This class only provides indexing and counting functionalities for the
/// sparse grid. The actual data stored on grid nodes is implemented in
/// grid_data.h.
///
/// @note For transfer purpose, the actual data that grid nodes should carry are
/// masses, momentums/velocities, and forces.

template <typename T>
class SparseGrid {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SparseGrid);

  /// Creates a SparseGrid with grid node spacing h.
  /// @pre h > 0.
  explicit SparseGrid(double h) : h_(h), Dp_inv_(4.0 / h / h) {}

  /// Preallocates memory for the sparse grid, based on a guess capacity of the
  /// upper bound of the number of active grid nodes.
  void Reserve(size_t capacity) {
    map_from_1d_to_3d_.reserve(capacity);
    map_from_3d_to_1d_.reserve(capacity);
    batch_sizes_.reserve(capacity);
  }

  /// Given a list of particle positions, first identifies and marks the active
  /// grid nods, then sorts them lexicographically. It also computes the (3d)
  /// batch index for each particle position, and stores in batch_indices. The
  /// (3d) batch index of a position is the 3d index of a node {i, j, k} whose
  /// distance to this position is the shortest among all grid nodes {(i,
  /// j, k) ⋅ h | i, j, k ∈ ℤ}.
  /// @pre batch_indices->size() == positions.size()
  /// @note User should call this function to update the SparseGrid whenever the
  /// underlying particle positions change, by *only* calling
  /// this.Update(particles.positions(), &particles.GetMutableBatchIndices()).
  /// All other functionalities of this class will then be up-to-date.
  void Update(const std::vector<Vector3<T>>& positions,
              std::vector<Vector3<int>>* batch_indices);

  size_t num_active_nodes() const { return map_from_1d_to_3d_.size(); }

  double h() const { return h_; }

  double Dp_inv() const { return Dp_inv_; }

  const std::vector<size_t>& batch_sizes() const { return batch_sizes_; }

  size_t GetBatchSizeAtNode(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes());
    return batch_sizes_[index_1d];
  }

  /// Returns the position at node indexed by (i, j, k), where we denote {i, j,
  /// k} = index_3d.
  /// @note The positions is (ih, jh, kh) by
  /// construction. This does NOT guarantee that index_3d is an active node.
  Vector3<T> GetPositionAt(const Vector3<int>& index_3d) const {
    return Vector3<T>(index_3d(0) * h_, index_3d(1) * h_, index_3d(2) * h_);
  }

  /// Checks whether the grid node at position (ih, jh, kh) is active or not,
  /// where we denote {i, j, k} = index_3d.
  bool IsActive(const Vector3<int>& index_3d) const {
    return map_from_3d_to_1d_.count(index_3d) == 1;
  }

  /// Reduces an 3D index_3d = {i, j, k} index to its corresponding 1D index on
  /// the sparse grid.
  /// @throw std::exception if (ih, jh, kh) is not an active node.
  size_t Reduce3DIndex(const Vector3<int>& index_3d) const {
    return map_from_3d_to_1d_.at(index_3d);
  }

  /// Expands a lexicographical index_1d to its corresponding 3D index (i,j,k).
  /// @pre index_1d < num_active_nodes()
  const Vector3<int>& Expand1DIndex(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes());
    return map_from_1d_to_3d_[index_1d];
  }

  /// Calculates the sparsity pattern of the hessian matrix d²e/dv² where e is
  /// the elastic energy and v is grid velocities. Each grid node is interating
  /// with its 124 neighbors. result[i] is a std::vector<int> that stores grid
  /// nodes that are interacting with node i and whose indices are larger than
  /// or equal to i, as the hessian is a symmetric matrix.
  /// <!-- TODO(zeshunzong) Documentation can be elaborated. Testing will be
  /// deferred when we check hession -->
  std::vector<std::vector<int>> CalcGridHessianSparsityPattern() const;

 private:
  /// Removes all existing active grid nodes.
  void ClearActiveNodes();

  /// Sorts active grid nodes lexicographically.
  void SortActiveNodes();

  /// For each position x in positions, computes the 3d grid index {i, j, k}
  /// such that (ih, jh, kh) is the closet grid node to position x. When the
  /// position x is the position of a particle, the corresponding 3d grid index
  /// {i, j, k} is called the batch index of this particle.
  /// @pre batch_indices->size() = positions.size()
  void ComputeBatchIndices(const std::vector<Vector3<T>>& positions,
                           std::vector<Vector3<int>>* batch_indices);

  /// Resets the counter for the number of particles in each batch
  void ResetBatchSizes() {
    batch_sizes_.resize(num_active_nodes());
    std::fill(batch_sizes_.begin(), batch_sizes_.end(), 0);
  }

  double h_{};

  /// This is the matrix Dₚ⁻¹ as in equation (173) in
  /// https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf. This
  /// is a diagonal matrix with all diagonal elements being the same. We thus
  /// store it as a scalar -- the common diagonal entry. For the quadratic
  /// interpolaton we use, Dₚ⁻¹ = (0.25 ⋅ h² ⋅ I) ⁻¹, so we hard-code it as
  /// Dp_inv_ = 4/h².
  double Dp_inv_{};
  std::vector<Vector3<int>> map_from_1d_to_3d_;
  std::unordered_map<Vector3<int>, size_t> map_from_3d_to_1d_;

  // A vector holding the number of particles inside each batch,
  // size=num_active_grids

  //           o ---- o ---- o ---- o ---- o ---- o
  //           |      |      |      |      |      |
  //           |   ppp|ppp  l|l     |    tt| ttzzz|
  //           o ---- o ---- o ---- o ---- o ---- o
  //           |   ppp|ppp   |ll    |   ttt|tt  zz|
  //           |   xxx|xxxuuu|uuu kk|kkk   |      |
  //           o ---- o ---- o ---- # ---- o ---- o
  //           |   xxx|xxxuuu|uuu k |  k   |      |
  //           |   yyy|yyy   |   qqq|q q   |      |
  //           o ---- * ---- o ---- o ---- o ---- o
  //
  // For computation purpose, particles clustered around one grid node are
  // classified into one batch (the batch is marked by their center grid node).
  // See the schematic illustration above. Each particle belongs to *exactly*
  // one batch, and there will always be num_active_nodes() batches (some of
  // them may be empty). batch_sizes_[i] records the number of particles in its
  // batch, i.e. the number of particles clustered around grid node with global
  // lexicographical index i. For instance, batch_sizes_[#] = 7. The sum of all
  // batch sizes should be the same as total number of particles.
  // @note batch_sizes_.size() = num_active_nodes()
  std::vector<size_t> batch_sizes_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
