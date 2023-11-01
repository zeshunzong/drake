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
 * Each particle p has its own PadSplatter. The attributes stored in this class
 * are up-to-date as long as the position of particle p does not change.
 * Throughout this class we refer to `this` particle as the particular particle
 * that is associated with `this` splatter.
 */
template <typename T>
class PadSplatter {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PadSplatter);

  PadSplatter() {}

  /**
   * Assuming quadratic interpolation, computes 27 weights and weight gradients
   * for the b-spline interpolation function centered at the 27 grid nodes
   * neighboring to `this` particle, evaluated at `this` particle. It also
   * records the global indices of the 27 grid nodes neighboring to `this`
   * particle. Data stored in this class will be marked as up-to-date and can be
   * used for transferring.
   */
  void Update(const Vector3<int>& particle_batch_index,
              const Vector3<T>& particle_position, const SparseGrid<T>& grid) {
    // lexicographically setup the 27 neighbor nodes
    int node_index_local;
    Vector3<int> node_index_3d;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // loop over 27 neighbor nodes
          // node_index_local is from 0 to 26
          node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);  // 0-26

          // node_index_3d is {x, y, z} such that {xh, yh, zh} is the position
          // of this node
          node_index_3d = particle_batch_index + Vector3<int>{a, b, c};
          pad_nodes_global_indices_[node_index_local] =
              grid.Reduce3DIndex(node_index_3d);

          const internal::BSpline<T> bpline =
              internal::BSpline<T>(grid.h(), grid.GetPositionAt(node_index_3d));
          const auto [wip, dwip] =
              bpline.ComputeValueAndGradient(particle_position);
          weights_[node_index_local] = wip;
          weight_gradients_[node_index_local] = dwip;
        }
      }
    }
  }

  /**
   * Splats the mass and velocity of the particle that corresponds to `this`
   * splatter to local_pad, which is a container of information on the 27
   * neighbor grid nodes. The 27 neighbor grid nodes in local_pad are assumed to
   * be ordered lexicographically.
   * @param[in] particle_batch_index the 3d batch index of `this` particle.
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
   * @param[out] local_pad       a pad stores info on the 27 neighbor nodes,
   * states of `this` particle is accumulated onto this local_pad They are the
   * (3D) adjacent neighbor nodes of the center node with index batch_index. We
   * follow the APIC transfer routine as in equations (172), (173), (189) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
   * <-- TODO(zeshunzong): consider removing the dependency on SparseGrid -->
   */
  void SplatToPad(const Vector3<int>& particle_batch_index,
                  const SparseGrid<T>& grid, const T& m_p,
                  const Vector3<T>& x_p, const Vector3<T>& v_p,
                  const Matrix3<T>& C_p, const T& V0_p, const Matrix3<T>& P_p,
                  const Matrix3<T>& FE_p, Pad<T>* local_pad) const {
    int node_index_local;
    Vector3<int> node_index_3d;

    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // loop over 27 neighbor nodes
          // node_index_local is from 0 to 26
          node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);  // 0-26

          // node_index_3d is {x, y, z} such that {xh, yh, zh} is the position
          // of this node
          node_index_3d = particle_batch_index + Vector3<int>{a, b, c};

          // eqn (172)
          (*local_pad).pad_masses[node_index_local] +=
              weights_[node_index_local] * m_p;
          // eqn (173)
          (*local_pad).pad_velocities[node_index_local] +=
              weights_[node_index_local] * m_p *
              (v_p + C_p * (grid.GetPositionAt(node_index_3d) - x_p));

          // eqn (188)
          (*local_pad).pad_forces[node_index_local] +=
              -V0_p * P_p * FE_p.transpose() *
              weight_gradients_[node_index_local];
        }
      }
    }
  }

  /**
   * transfer the 27 local grid velocities to particle
   * more tbd...
   */
  Vector3<T> AccumulateToParticle(const Pad<T>& local_pad) {
    // For each grid node affecting the current particle
    Vector3<T> vp_new_i;
    Vector3<T> vp_new = Vector3<T>::Zero();
    int node_index_local;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
          vp_new_i = local_pad.pad_velocities[node_index_local] *
                     weights_[node_index_local];
          // v_p^{n+1} = \sum v_i^{n+1} N_i(x_p)
          vp_new += vp_new_i;
        }
      }
    }

    return vp_new;
  }

 private:
  std::array<size_t, 27> pad_nodes_global_indices_{};

  // weights_[i] = w_ip
  std::array<T, 27> weights_{};
  // weight_gradients_[i] = dw_ip
  std::array<Vector3<T>, 27> weight_gradients_{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
