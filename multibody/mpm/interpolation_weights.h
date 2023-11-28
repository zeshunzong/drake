#pragma once

#include <array>
#include <memory>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/internal/b_spline.h"
#include "drake/multibody/mpm/internal/math_utils.h"
#include "drake/multibody/mpm/pad.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct ParticleVBGradV {
  Vector3<T> v;         // new particle velocity
  Matrix3<T> B_matrix;  // new B_matrix
  Matrix3<T> grad_v;    // grad_v for updating F
};

/**
 * A class performing the interpolation operations between one particle and its
 * neighboring nodes. In our implementation of MPM, we adopt the quadratic
 * B-spline. Thus, for a particular particle p, the interpolation weights w_ip
 * will only be non-zero for the B-spline kernels centered at its 27 neighbor
 * grid nodes i. Grid nodes further away live outside the support. As a result,
 * all operations are performed for the 27 neighbor nodes.
 *                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
 *                ⋅    __⋅__    ⋅
 *                ⋅   ▏  ⋅  ▕   ⋅
 *                o ⋅ ▏⋅ x ⋅▕ ⋅ o
 *                ⋅   ▏p_⋅__▕   ⋅
 *                ⋅      ⋅      ⋅
 *                o ⋅⋅⋅⋅ o ⋅⋅⋅⋅ o
 * A schematic graph in 2D is shown above. For particle p, the grid nodes that
 * are in support are denoted with `o`. `x` is also one of them, and is further
 * called the base node. In 2D there are nine of them. Note that the 27 nodes
 * form a cube centered around the particle p. The 27 nodes are given a local
 * lexicographical index, following
 * mpm::internal::CompareIndex3DLexicographically().
 */
template <typename T>
class InterpolationWeights {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(InterpolationWeights);
  /**
   * All weights and weight gradients are initialized to be zero. The weights
   * will be invalid until Reset() is called.
   */
  InterpolationWeights() = default;

  /**
   * Given the particle's position, computes the weights and weight gradients
   * for all 27 neighbor nodes. When the particle's position changes, the
   * weights need to be updated as well.
   */
  void Reset(const Vector3<T>& position, const Vector3<int>& base_node,
             double h) {
    int node_index_local;
    Vector3<int> node_index_3d;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // loop over 27 neighbor nodes
          // node_index_local is from 0 to 26
          node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);  // 0-26
          node_index_3d = base_node + Vector3<int>{a, b, c};

          const internal::BSpline<T> bpline = internal::BSpline<T>(
              h, internal::ComputePositionFromIndex3D(node_index_3d, h));
          const auto [wip, dwip] = bpline.ComputeValueAndGradient(position);
          weights_[node_index_local] = wip;
          weight_gradients_[node_index_local] = dwip;
        }
      }
    }
  }

  /**
   * Accumulates the mass, velocity, and stress from the p-th particle onto a
   * local pad, which stores the grid states for 27 neighbor nodes to the p-th
   * particle.
   * @note data is *accumulated* to pad.
   * @note particles in the same batch should accumulate to the same pad. We
   * follow the APIC transfer routine as in equations (172), (173), (189) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
   */
  void SplatParticleDataToPad(const T& m_p, const Vector3<T>& x_p,
                              const Vector3<T>& v_p, const Matrix3<T>& C_p,
                              const T& V0_p, const Matrix3<T>& P_p,
                              const Matrix3<T>& FE_p,
                              const Vector3<int>& base_node, double h,
                              Pad<T>* pad) const {
    (*pad).base_node = base_node;
    int node_index_local;
    Vector3<int> node_index_3d;
    Vector3<T> node_position;

    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          // loop over 27 neighbor nodes
          // node_index_local is from 0 to 26
          node_index_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);  // 0-26
          node_index_3d = base_node + Vector3<int>{a, b, c};

          // eqn (172), mass
          (*pad).masses[node_index_local] += weights_[node_index_local] * m_p;
          // eqn (173), momentum
          (*pad).momentums[node_index_local] +=
              weights_[node_index_local] * m_p *
              (v_p +
               C_p * (internal::ComputePositionFromIndex3D(node_index_3d, h) -
                      x_p));
          // eqn (188), force
          (*pad).forces[node_index_local] +=
              -V0_p * P_p * FE_p.transpose() *
              weight_gradients_[node_index_local];
        }
      }
    }
  }

  /**
   * Given the grid data on 27 (3by3by3 cube) nodes stored in batch_pad that are
   * in the support of p-th particle, computes and returns the new velocity, new
   * B_matri, and new velocity gradient of the p-th particle.
   */
  ParticleVBGradV<T> AccumulateFromBatchPad(const Vector3<T>& x_p,
                                            const BatchPad<T>& batch_pad) {
    const std::array<Vector3<T>, 27>& pad_positions = batch_pad.positions;
    const std::array<Vector3<T>, 27>& pad_velocities = batch_pad.velocities;
    Vector3<T> vi_new;
    Vector3<T> vp_new = Vector3<T>::Zero();
    Matrix3<T> Bp_new = Matrix3<T>::Zero();
    Matrix3<T> grad_vp_new = Matrix3<T>::Zero();
    int idx_local;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);

          vi_new = pad_velocities[idx_local] * weights_[idx_local];
          // v_p^{n+1} = \sum v_i^{n+1} N_i(x_p)
          vp_new += vi_new;

          // B_p^{n+1} = \sum v_i^{n+1}*(x_i - x_p^n)^T N_i(x_p)
          Bp_new += vi_new * (pad_positions[idx_local] - x_p).transpose();

          // Accumulate grad_vp_new: F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
          grad_vp_new += vi_new * weight_gradients_[idx_local].transpose();
        }
      }
    }
    return {vp_new, Bp_new, grad_vp_new};
  }

 private:
  std::array<T, 27> weights_{};
  std::array<Vector3<T>, 27> weight_gradients_{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
