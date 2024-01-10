#pragma once

#include <unordered_map>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/internal/hashing_utils.h"
#include "drake/multibody/mpm/internal/mass_and_momentum.h"
#include "drake/multibody/mpm/internal/math_utils.h"
#include "drake/multibody/mpm/pad.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * Implements a (3D) sparse grid that serves as the background Eulerian grid in
 * the Material Point Method. In MPM, a Cartesian grid is used to update
 * particle states. Particles reside in the grid as (schematically in 2D)
 *           o - o - o - o - o - o - o
 *           |   |   |   |x  |   |   |
 *           o - o - o - o - o - o - o
 *           |   |x x|x  |xx |   |   |
 *           o - o - o - o - o - o - o
 *           |   |x  |   |   |   |   |
 *           o - o - o - o - o - o - o
 *           |   |   |   |   |   |   |
 *           o - o - o - o - o - o - o
 *               Fig 1. Dense grid.
 *
 * where     <-h-> h > 0 is the spacing between grid nodes,
 *           o denotes a grid node,
 * and       x denotes a particle residing in the grid.
 * In MPM, the interpolation function we use (see b_spline.h) has a finite
 * support. Thus, each particle x only interacts with its "neighbor" grid
 * nodes. Therefore, a union of all the "neighbor" grid nodes would suffice for
 * the MPM algorithm. They are denoted as the "active grid nodes" for a sparse
 * grid. Schematically, those active grid nodes are
 *                       o - o
 *                       |x  |
 *               o - o - o - o
 *               |x x|x  |xx |
 *               o - o - o - o
 *               |x  |
 *               o - o
 *
 *               Fig 2. Sparse grid.
 *
 * @note This is only a schematic illustration. The actual number of active
 * grid nodes also depends on the interpolation function we choose.
 *
 * @note Currently only quadratic B-spline is supported, where the support is
 * [-1.5h, 1.5h].
 *
 * Given grid node spacing h, the grid nodes for the dense grid (c.f. Fig 1)
 * are characterized by the set D = {(i, j, k) ⋅ h | i, j, k ∈ ℤ}.
 * Correspondingly, the grid nodes for a sparse grid forms a subset S ⊂ D such
 * that each o ∈ S is a neighbor node of at least one particle x.
 *
 * Under quadratic B-spline, each particle will activate its 27 neighbor grid
 * nodes in 3D (9 in 2D). See the illustration below.
 *
 *                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
 *                ⋅    __⋅__    ⋅
 *                ⋅   ▏  ⋅  ▕   ⋅
 *                o ⋅ ▏⋅ X ⋅▕ ⋅ o
 *                ⋅   ▏p_⋅__▕   ⋅
 *                ⋅      ⋅      ⋅
 *                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
 * Fig 3. Neighbor grid nodes activated by a particle p.
 *
 * For a grid node i, any particle p that falls in an equidistant bounding box
 * with width 2×1.5h centered at i will result in a non-zero interpolation
 * function and thus interact with p. We can safely discard the boundary case
 * (i.e. a particle falls exactly onto the boundary of the bounding box) as the
 * interpolation function is nil at 1.5h. Equivalently, for a particle p, an
 * equidistant bounding box with width 2×1.5h centered at p will cover all grid
 * nodes whom p will interact with. In 1D this includes 3 nodes; in 2D this
 * includes 9 nodes (c.f. Fig 3); and in 3D this includes 27 nodes. In
 * particular, for a particle p, we can always find its nearest grid node
 * (denoted X in Fig 3). Write the 3d index of node X as {i,j,k}. Then nodes
 * indexed by {i±1, j±1, k±1} will all interact with particle p.
 *
 * @note Following the above observation, we define {i,j,k} to be the base node
 * for particle p, where (ih, jh, kh) is the position of grid node X, c.f.
 * Fig 3.
 *
 * To summarize, we implement the MPM in 3D with quadratic B spline
 * interpolation function, and thus one particle will activate its 27 neighbor
 * grids.
 *
 * @note The active grid nodes depend on the positions of particles. When the
 * particles move to new positions, the sparse grid is updated as some existing
 * active nodes cease to be active, while some new neighbor nodes join.
 */

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

  size_t num_active_nodes() const { return map_1d_to_3d_.size(); }

  double h() const { return h_; }

  /**
   * Given a list of base nodes, mark all their neighbors as active nodes, c.f.
   * Fig. 3. map_3d_to_1d_ and map_1d_to_3d_ are updated accordingly.
   * A node is considered active if and only if it's a neighbor of at least one
   * node in the *current*base_nodes.
   */
  void MarkActiveNodes(const std::vector<Vector3<int>>& base_nodes);

  /**
   * Checks whether the grid node at position (ih, jh, kh) is active or not,
   * where we denote {i, j, k} = index_3d.
   */
  bool IsActive(const Vector3<int>& index_3d) const {
    return map_3d_to_1d_.count(index_3d) == 1;
  }

  /**
   * Given 3d index, returns the corresponding 1d index
   * @throw exception if (ih, jh, kh) is not an active node.
   */
  size_t To1DIndex(const Vector3<int>& index_3d) const {
    return map_3d_to_1d_.at(index_3d);
  }

  /**
   * Given 1d index, returns the corresponding 3d index.
   * @pre index_1d < num_active_nodes()
   */
  const Vector3<int>& To3DIndex(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < num_active_nodes());
    return map_1d_to_3d_[index_1d];
  }

  /**
   * Computes the states on grid by adding all data stored on pads.
   * @pre Pads doesn't contain data into non-active region of the grid.
   */
  void GatherFromP2gPads(const std::vector<P2gPad<T>>& p2g_pads,
                         GridData<T>* grid_data) const;

  /**
   * Computes the forces on grid by adding stresses stored on pads.
   * @note this is a partial version of GatherFromP2gPads(), where only grid
   * forces are computed.
   * @pre Pads doesn't contain data into non-active region of the grid.
   */
  void GatherForceFromP2gPads(const std::vector<P2gPad<T>>& p2g_pads,
                              std::vector<Vector3<T>>* grid_forces) const;

  /**
   * Computes the mass and momentum of the body embedded in this grid, by
   * summing over all active grid nodes.
   */
  internal::MassAndMomentum<T> ComputeTotalMassMomentum(
      const GridData<T>& grid_data) const;

 private:
  double h_{};

  // maps the 3d index (i,j,k) of a node to its 1d index in the memory
  std::unordered_map<Vector3<int>, size_t> map_3d_to_1d_{};

  // maps l-th node in memory to its corresponding 3d index
  std::vector<Vector3<int>> map_1d_to_3d_{};

  // @note: if node (i,j,k) -- that is, node at position (i*h_, j*h_, k*h_) --
  // has 1d index l, then map_1d_to_3d_[l] = {i,j,k} and
  // map_3d_to_1d_.at({i,j,k}) = l.
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
