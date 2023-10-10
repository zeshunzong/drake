#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A Particles class holding particle states as several std::vectors.
 * Each particle carries its own position, velocity, mass, volume, etc.
 * 
 * The Material Point Method (MPM) consists of a set of particles (implemented in this class) and a background Eulerian grid (implemented in sparse_grid.h).
 * At each time step, particle mass and momentum are transferred to the grid nodes via a B-spline interpolation function
 * (implemented in internal::b_spline.h). 
 */
template <typename T>
class Particles {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Particles);

  /**
   * Creates a Particles container with 0 particle. All std::vector<> are set
   * to length zero. This, working together with AddParticle(), should be the
   * default version.
   */
  Particles();

  /**
   * Creates a Particles container with num_particles particles.
   * Length of each std::vector is set to num_particles, but zero-initialized.
   * This will be used sampling info is loaded externally.
   */
  explicit Particles(size_t num_particles);

  // Adds (appends) a particle into Particles with given properties.
  // TODO(zeshunzong): More attributes will come later.
  void AddParticle(const Vector3<T>& position, const Vector3<T>& velocity,
                   const T& mass, const T& reference_volume,
                   const Matrix3<T>& deformation_gradient,
                   const Matrix3<T>& B_matrix);

  // Permutes all states in Particles with respect to the index set
  // new_order. e.g. given new_order = [2; 0; 1], and the original
  // particle states are denoted by [p0; p1; p2]. The new particles states
  // after calling Reorder() will be [p2; p0; p1]
  // @pre new_order is a permutation of [0, ..., num_particles-1]
  void Reorder(const std::vector<size_t>& new_order);

  /**
   * Advects each particle's position x_p by dt*v_p, where v_p is particle's
   * velocity.
   */
  void AdvectParticles(double dt) {
    for (size_t p = 0; p < num_particles_; ++p) {
      positions_[p] += dt * velocities_[p];
    }
  }

  size_t num_particles() const { return num_particles_; }

  // Disambiguation:
  // positions: a std::vector holding position of all particles.
  // position:  the position of a particular particle. This shall always be
  // associated with a particle index p.
  // This naming rule applies to all class attributes.

  const std::vector<Vector3<T>>& positions() const { return positions_; }
  const std::vector<Vector3<T>>& velocities() const { return velocities_; }
  const std::vector<T>& masses() const { return masses_; }
  const std::vector<T>& reference_volumes() const { return reference_volumes_; }

  // Note: one must know beforehand if the data stored is F_trial or F_elastic.
  const std::vector<Matrix3<T>>& deformation_gradients() const {
    return deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& B_matrices() const { return B_matrices_; }

  /**
   * Returns the position of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Vector3<T>& GetPositionAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return positions_[p];
  }

  /**
   * Returns the velocity of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Vector3<T>& GetVelocityAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return velocities_[p];
  }

  /**
   * Returns the mass of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const T& GetMassAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return masses_[p];
  }

  /**
   * Returns the reference volume of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const T& GetReferenceVolumeAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return reference_volumes_[p];
  }

  /**
   * Returns the deformation gradient of p-th particle.
   * @pre 0 <= p < num_particles()
   * @note one must know beforehand if the data stored is F_trial or F_elastic.
   */
  const Matrix3<T>& GetDeformationGradientAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return deformation_gradients_[p];
  }

  /**
   * Returns the B_matrix of p-th particle. B_matrix is part of the affine
   * momentum matrix C as
   * v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetBMatrixAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return B_matrices_[p];
  }

  /**
   * Sets the position at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetPositionAt(size_t p, const Vector3<T>& position) {
    DRAKE_ASSERT(p < num_particles_);
    positions_[p] = position;
  }

  /**
   * Sets the velocity at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetVelocityAt(size_t p, const Vector3<T>& velocity) {
    DRAKE_ASSERT(p < num_particles_);
    velocities_[p] = velocity;
  }

  /**
   * Sets the deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   * @note one should know if F_in is F_trial or F_elastic.
   */
  void SetDeformationGradient(size_t p, const Matrix3<T>& F_in) {
    DRAKE_ASSERT(p < num_particles_);
    deformation_gradients_[p] = F_in;
  }

  /**
   * Sets the B_matrix at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetBMatrix(size_t p, const Matrix3<T>& B_matrix) {
    DRAKE_ASSERT(p < num_particles_);
    B_matrices_[p] = B_matrix;
  }

  /**
   * For the node_local-th neighbor (out of a total of 27) grid node of particle
   * p, stores its global index in sparse_grid given by node_global.
   * @pre p < num_particles()
   * @pre node_local < 27
   * @pre node_global < sparse_grid.num_active_nodes()
   */
  void SetNeighborNodeGlobalIndex(size_t p, size_t node_local,
                                  size_t node_global) {
    DRAKE_ASSERT(p < num_particles_);
    DRAKE_ASSERT(node_local < 27);
    neighbor_grid_nodes_global_indices_[p][node_local] = node_global;
  }

  /**
   * Returns the global index (in sparse_grid) of the node_local-th neighbor
   * grid node of particle p.
   * @pre p < num_particles()
   * @pre node_local < 27
   */
  size_t GetNeighborNodeGlobalIndex(size_t p, size_t node_local) const {
    return neighbor_grid_nodes_global_indices_[p][node_local];
  }

  /**
   * Given particle index p and node_local i, stores w_ip =
   * Nᵢ(xₚ).
   * @pre p < num_particles()
   * @pre node_local < 27
   */
  void SetWeightAtParticleAndNeighborNode(size_t p, size_t node_local,
                                          const T& w_ip) {
    DRAKE_ASSERT(p < num_particles_);
    DRAKE_ASSERT(node_local < 27);
    w_ip_neighbor_nodes_[p][node_local] = w_ip;
  }

  /**
   * Given particle index p and node_local i, stores dw_ip =
   * ∇Nᵢ(xₚ).
   * @pre p < num_particles()
   * @pre node_local < 27
   */
  void SetWeightGradientAtParticleAndNeighborNode(size_t p, size_t node_local,
                                                  const Vector3<T>& dw_ip) {
    DRAKE_ASSERT(p < num_particles_);
    DRAKE_ASSERT(node_local < 27);
    dw_ip_neighbor_nodes_[p][node_local] = dw_ip;
  }

  /**
   * Given particle index p and node_local i, returns the stored
   * Nᵢ(xₚ).
   * @pre p < num_particles()
   * @pre node_local < 27
   * @note the stored Nᵢ(xₚ) is meaningless when xₚ changes.
   */
  const T& GetWeightAtParticleAndNeighborNode(size_t p,
                                              size_t node_local) const {
    DRAKE_ASSERT(p < num_particles_);
    DRAKE_ASSERT(node_local < 27);
    return w_ip_neighbor_nodes_[p][node_local];
  }

  /**
   * Given particle index p and node_local i, returns the stored
   * ∇Nᵢ(xₚ).
   * @pre p < num_particles()
   * @pre node_local < 27
   * @note the stored ∇Nᵢ(xₚ) is meaningless when xₚ changes.
   */
  const Vector3<T>& GetWeightGradientAtParticleAndNeighborNode(
      size_t p, size_t node_local) const {
    DRAKE_ASSERT(p < num_particles_);
    DRAKE_ASSERT(node_local < 27);
    return dw_ip_neighbor_nodes_[p][node_local];
  }

 private:
  size_t num_particles_ = 0;
  std::vector<Vector3<T>> positions_{};
  std::vector<Vector3<T>> velocities_{};
  std::vector<T> masses_{};
  std::vector<T> reference_volumes_{};
  // This may be either F_trial or F_elastic, depending on the context.
  std::vector<Matrix3<T>> deformation_gradients_{};

  // The affine matrix B_p in APIC
  // B_matrix is part of the affine momentum matrix C as
  // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
  std::vector<Matrix3<T>> B_matrices_{};

  // Each particle p lives in the support of (quadratic) BSpline functions
  // centered at 27 neighbor grid nodes. neighbor_grid_nodes_global_indices_[p]
  // stores the global indices of the 27 neighbor grid nodes.
  // w_ip_neighbor_nodes_[p] stores the weights Nᵢ(xₚ) for i ∈ [27 neighbor grid
  // nodes]. dw_ip_neighbor_nodes_[p] stores the weight gradients ∇Nᵢ(xₚ) for i
  // ∈ [27 neighbor grid nodes].
  std::vector<std::array<size_t, 27>> neighbor_grid_nodes_global_indices_{};
  std::vector<std::array<T, 27>> w_ip_neighbor_nodes_{};
  std::vector<std::array<Vector3<T>, 27>> dw_ip_neighbor_nodes_{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
