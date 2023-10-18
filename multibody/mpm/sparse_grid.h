#pragma once

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/internal/hashing_utils.h"
#include "drake/multibody/mpm/internal/math_utils.h"
#include "drake/multibody/mpm/internal/total_mass_momentum.h"

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
/// @note The lexicographical ordering looks like (0,0,0), (0,0,1), (0,1,0),
/// (0,1,1), (1,0,0), ...

template <typename T>
class SparseGrid {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SparseGrid);
  SparseGrid() = delete;

  /**
   * Creates a SparseGrid with grid node spacing h.
   * @pre h > 0.
   */
  explicit SparseGrid(double h);

  /**
   * Preallocates memory for the sparse grid, based on a guess capacity of the
   * upper bound of the number of active grid nodes.
   */
  void Reserve(size_t capacity);

  size_t num_active_nodes() const { return num_active_nodes_; }

  double h() const { return h_; }

  /**
   * Checks whether the grid node at position (ih, jh, kh) is active or not,
   * where we denote {i, j, k} = index_3d.
   */
  bool IsActive(const Vector3<int>& index_3d) const {
    return index_map_.count(index_3d) == 1;
  }

  /**
   * Reduces an 3D index_3d = {i, j, k} index to its corresponding 1D index on
   * the sparse grid.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  size_t Reduce3DIndex(const Vector3<int>& index_3d) const {
    return index_map_.at(index_3d);
  }

  /**
   * Expands a lexicographical index_1d to its corresponding 3D index (i,j,k).
   * @pre index_1d < num_active_nodes()
   */
  const Vector3<int>& Expand1DIndex(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    return active_nodes_[index_1d];
  }

  const Vector3<double>& gravitational_acceleration() const {
    return gravitational_acceleration_;
  }

  void SetGravity(const Vector3<double>& g) { gravitational_acceleration_ = g; }

  const std::vector<Vector3<T>>& velocities() const { return velocities_; }

  const std::vector<T>& masses() const { return masses_; }

  const std::vector<Vector3<T>>& forces() const { return forces_; }

  const std::vector<size_t>& batch_sizes() const { return batch_sizes_; }

  /**
   * For each position x in positions, computes the 3d grid index {i, j, k} such
   * that (ih, jh, kh) is the closet grid node to position x. When the position
   * x is the position of a particle, the corresponding 3d grid index {i, j, k}
   * is called the batch index of this particle.
   * @note the returned std::vector has the same size as positions
   */
  std::vector<Vector3<int>> ComputeBatchIndices(
      const std::vector<Vector3<T>>& positions) const;

  /**
   * Adds the node indexed by (i, j, k), where we denote {i, j,
   * k} = index_3d, into the collection of active grid nodes.
   * @note Calling this function will destroys the lexicographical ordering of
   * active grid nodes.
   * <!-- TODO(zeshunzong) Consider making it private. -->
   */
  void AppendActiveNode(const Vector3<int>& index_3d);

  /**
   * Removes all existing active grid nodes.
   * <!-- TODO(zeshunzong) Consider making it private. -->
   */
  void ClearActiveNodes();

  /**
   * Sorts active grid nodes lexicographically.
   * <!-- TODO(zeshunzong) Consider making it private. -->
   */
  void SortActiveNodes();

  /**
   * Given the positions of a list of particles, activates their neighbor grid
   * nodes and sorts them. It also resizes masses_, forces_, and velocities_ to
   * num_active_nodes(), i.e. the total number of non-repeated neighbor grid
   * nodes nodes.
   */
  void UpdateActiveNodesFromParticlePositions(
      const std::vector<Vector3<T>>& positions);

  /**
   * Overwrites all active grid node velocities to be v_in.s
   * @pre v_in should be velocities of the same existing active grid nodes (i.e.
   * same size, same ordering).
   */
  void SetVelocities(const std::vector<Vector3<T>>& v_in) {
    DRAKE_DEMAND(velocities_.size() == v_in.size());
    velocities_ = v_in;
  }

  /**
   * Returns the position at node indexed by (i, j, k), where we denote {i, j,
   * k} = index_3d.
   * @note The positions is (ih, jh, kh) by
   * construction. This does NOT guarantee that index_3d is an active node.
   */
  Vector3<T> GetPositionAt(const Vector3<int>& index_3d) const {
    return Vector3<T>(index_3d(0) * h_, index_3d(1) * h_, index_3d(2) * h_);
  }

  // Disambiguation:
  // Node velocity, mass and force can be accessed by either index_3d or
  // index_1d. See class document for their conversions.

  /**
   * Returns the velocity at node indexed by (i, j, k), where we denote {i, j,
   * k} = index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  const Vector3<T>& GetVelocityAt(const Vector3<int>& index_3d) const {
    return velocities_[Reduce3DIndex(index_3d)];
  }

  /**
   * Returns the velocity at node indexed by index_1d.
   * @pre index_1d < num_active_nodes().
   */
  const Vector3<T>& GetVelocityAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    return velocities_[index_1d];
  }

  /**
   * Returns the mass at node indexed by (i, j, k), where we denote {i, j, k} =
   * index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  const T& GetMassAt(const Vector3<int>& index_3d) const {
    return masses_[Reduce3DIndex(index_3d)];
  }

  /**
   * Returns the mass at node indexed by index_1d.
   * @pre index_1d < num_active_nodes().
   */
  const T& GetMassAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    return masses_[index_1d];
  }

  /**
   * Returns the force at node indexed by (i, j, k), where we denote {i, j, k} =
   * index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  const Vector3<T>& GetForceAt(const Vector3<int>& index_3d) const {
    return forces_[Reduce3DIndex(index_3d)];
  }

  /**
   * Returns the force at node indexed by index_1d.
   * @pre index_1d < num_active_nodes().
   */
  const Vector3<T>& GetForceAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    return forces_[index_1d];
  }

  /**
   * Sets the force at node indexed by (i, j, k), where we denote {i, j, k} =
   * index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  void SetForceAt(const Vector3<int>& index_3d, const Vector3<T>& force) {
    forces_[Reduce3DIndex(index_3d)] = force;
  }

  /**
   * Sets the force at node indexed by index_1d.
   * @pre index_1d < num_active_nodes().
   */
  void SetForceAt(size_t index_1d, const Vector3<T>& force) {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    forces_[index_1d] = force;
  }

  /**
   * Adds vel to velocity at node indexed by (i, j, k), where we denote {i, j,
   * k} = index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  void AccumulateVelocityAt(const Vector3<int>& index_3d,
                            const Vector3<T>& vel) {
    velocities_[Reduce3DIndex(index_3d)] += vel;
  }

  /**
   * Adds m to mass at node indexed by (i, j, k), where we denote {i, j, k} =
   * index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  void AccumulateMassAt(const Vector3<int>& index_3d, const T& m) {
    masses_[Reduce3DIndex(index_3d)] += m;
  }

  /**
   * Adds f to force at node indexed by (i, j, k), where we denote {i, j, k} =
   * index_3d.
   * @throw Out-of-range exception if (ih, jh, kh) is not an active node.
   */
  void AccumulateForce(const Vector3<int>& index_3d, const Vector3<T>& f) {
    forces_[Reduce3DIndex(index_3d)] += f;
  }

  /**
   * Zeros out grid masses, forces, and velocties.
   * @pre the size of three vectors must be equal to num_active_nodes()
   */
  void ResetMassesForcesVelocities();

  /**
   * Sets the size of the vectors storing grid masses, forces, and velocties to
   * be num_active_nodes().
   */
  void ResizeMassesForcesVelocities();

  /**
   * Converts grid momentums to grid velocities.
   * @note In P2G we will temporarily store the momentum m*v into velocites.
   */
  void ConvertMomentumsToVelocities() {
    for (size_t i = 0; i < num_active_nodes_; ++i) {
      velocities_[i] = velocities_[i] / masses_[i];
    }
  }

  /**
   * Updates grid velocity by vⁿ⁺¹ = vⁿ + dt*fⁿ/mⁿ where f is the stored grid
   * velocities and m is the stored grid masses.
   */
  void UpdateVelocity(double dt) {
    for (size_t i = 0; i < num_active_nodes_; ++i) {
      velocities_[i] += dt * forces_[i] / masses_[i];
    }
  }

  /**
   * Applies gravity to grid velocities.
   * v = v + dt * g
   */
  void ApplyGravity(double dt) {
    Vector3<T> dt_times_g = dt * gravitational_acceleration_;
    for (size_t i = 0; i < num_active_nodes_; ++i) {
      velocities_[i] += dt_times_g;
    }
  }

  /**
   * Returns the sum of mass, momentum and angular momentum of all active grid
   * nodes. For testing purpose only.
   */
  internal::TotalMassMomentum<T> ComputeTotalMassMomentum() const;

  /**
   * Calculates the sparsity pattern of the hessian matrix d²e/dv² where e is
   * the elastic energy and v is grid velocities. Each grid node is interating
   * with its 124 neighbors. result[i] is a std::vector<int> that stores grid
   * nodes that are interacting with node i and whose indices are larger than or
   * equal to i, as the hessian is a symmetric matrix.
   * <!-- TODO(zeshunzong) Documentation can be elaborated. Testing will be
   * deferred when we check hession -->
   */
  std::vector<std::vector<int>> CalcGridHessianSparsityPattern() const;

 private:
  // Returns the number particles that cluster around node indexed by (i, j, k),
  // where we denote {i, j, k} = index_3d. See document for batch_sizes_
  // @throw Out-of-range exception if (ih, jh, kh) is not an active node.
  size_t GetBatchSizeAt(const Vector3<int>& index_3d) const {
    return batch_sizes_[Reduce3DIndex(index_3d)];
  }

  // Returns the number particles that cluster around node indexed by index_1d.
  // @pre index_1d < num_active_nodes().
  size_t GetBatchSizeAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes_);
    return batch_sizes_[index_1d];
  }

  void ResetBatchSizes() {
    batch_sizes_.resize(num_active_nodes_);
    fill(batch_sizes_.begin(), batch_sizes_.end(), 0);
  }

  size_t num_active_nodes_;
  double h_;

  // A unordered map from the 3D physical index to the 1D index in the memory
  std::unordered_map<Vector3<int>, size_t> index_map_{};

  // Stores the lexicographical ordering of active grid nodes.
  // active_nodes_[ind] = {i,j,k} means that the grid node with position
  // {ih,jh,kh} has 1d lexicographical ind
  std::vector<Vector3<int>> active_nodes_{};

  // velocity, mass, and force at each active grid node.
  std::vector<Vector3<T>> velocities_{};
  std::vector<T> masses_{};
  std::vector<Vector3<T>> forces_{};

  Vector3<double> gravitational_acceleration_ = {0.0, 0.0, -9.8};

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
  std::vector<size_t> batch_sizes_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
