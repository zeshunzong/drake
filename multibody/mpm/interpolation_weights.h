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

 private:
  std::array<T, 27> weights_{};
  std::array<Vector3<T>, 27> weight_gradients_{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
