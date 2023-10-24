#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/multibody/mpm/internal/b_spline.h"
#include "drake/multibody/mpm/pad.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A class that stores interpolation weights and interpolation utilities for
 * transferring particle data to neighbor grid nodes. Here we only support a
 * quadratic b-spline interpolation.
 * Each particle p has its own PadSplatter. The attrbutes stored in this class
 * are up-to-date as long as the position of particle p does not change.
 * Throughout this class we refer to `this` particle as the particular particle
 * that is associated with `this` splatter.
 */
template <typename T>
class PadSplatter {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PadSplatter);

  /**
   * Splats the mass and velocity of the particle that corresponds to `this`
   * splatter to local_pad, which is a container of information on the 27
   * neighbor grid nodes. The 27 neighbor grid nodes in local_pad are assumed to
   * be ordered lexicographically.
   * @param[in] p                the index of `this` particle.
   * @param[in] sparse_grid      the underlying sparse grid for querying
   * neighbor nodes.
   * @param[in] m_p              the mass of `this` particle.
   * @param[in] x_p              the position of `this` particle.
   * @param[in] v_p              the velocity of `this` particle.
   * @param[in] C_p              the affine momentum of `this` particle.
   * @param[in] V0_p             the initial volume of `this` particle.
   * @param[in] P_p              the Kirchhoff stress of `this` particle.
   * @param[in] FE_p             the *elastic* deformation gradient of `this`
   * particle.
   * @param[out] local_pad       a pad stores info on the 27 neighbor nodes.
   * They are the (3D) adjacent neighbor nodes of the center node with index
   * batch_index.
   * We follow the APIC transfer routine as in equations (172), (173), (189) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
   */
  void SplatToPad(size_t p, const SparseGrid<T>& sparse_grid, const T& m_p,
                  const Vector3<T>& x_p, const Vector3<T>& v_p,
                  const Matrix3<T>& C_p, const T& V0_p, const Matrix3<T>& P_p,
                  const Matrix3<T>& FE_p, Pad<T>* local_pad) const {
    int node_idx_local;
    Vector3<int> node_index_3d;

    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // loop over 0 to 26
          node_idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
          node_index_3d =
              sparse_grid.GetParticleBatchIndexAt(p) + Vector3<int>{a, b, c};
          // eqn (172)
          (*local_pad).pad_masses[node_idx_local] +=
              weights_[node_idx_local] * m_p;
          // eqn (173)
          (*local_pad).pad_velocities[node_idx_local] +=
              weights_[node_idx_local] * m_p *
              (v_p + C_p * (sparse_grid.GetPositionAt(node_index_3d) - x_p));
          // eqn (188)
          (*local_pad).pad_forces[node_idx_local] +=
              -V0_p * P_p * FE_p.transpose() *
              weight_gradients_[node_idx_local];
        }
      }
    }
  }

 private:
  std::array<size_t, 27> pad_nodes_global_indices_{};

  // weights_[i] = w_ip
  std::array<T, 27> weights_{};
  std::array<Vector3<T>, 27> weight_gradients_{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
