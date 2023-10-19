#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

#include "drake/multibody/mpm/pad_splatter.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A Particles class holding particle states as several std::vectors.
 * Each particle carries its own position, velocity, mass, volume, etc.
 *
 * The Material Point Method (MPM) consists of a set of particles (implemented
 * in this class) and a background Eulerian grid (implemented in sparse_grid.h).
 * At each time step, particle mass and momentum are transferred to the grid
 * nodes via a B-spline interpolation function (implemented in
 * internal::b_spline.h).
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

  /**
   *  Adds (appends) a particle into Particles with given properties.
   * <!-- TODO(zeshunzong): More attributes will come later. -->
   * <!-- TODO(zeshunzong): Do we always start from rest shape? so F=I? -->
   */
  void AddParticle(const Vector3<T>& position, const Vector3<T>& velocity,
                   const T& mass, const T& reference_volume,
                   const Matrix3<T>& trial_deformation_gradient,
                   const Matrix3<T>& elastic_deformation_gradient,
                   const Matrix3<T>& B_matrix);

  /**
   * Permutes all states in Particles with respect to the index set new_order.
   * e.g. given new_order = [2; 0; 1], and the original particle states are
   * denoted by [p0; p1; p2]. The new particles states after calling Reorder()
   * will be [p2; p0; p1].
   * @pre new_order is a permutation of [0, ..., num_particles-1]
   * @note this algorithm uses index-chasing and might be O(n^2) in worst case.
   * <!-- TODO(zeshunzong): this algorithm is insanely fast for "simple"
   * permutations. A standard O(n) algorithm is implemented below in Reorder2().
   * We should decide on which one to choose once the whole MPM pipeline is
   * finished. -->
   * <!-- TODO(zeshunzong): May need to reorder more attributes as more
   * attributes are added. -->
   */
  void Reorder(const std::vector<size_t>& new_order);

  /**
   * Performs the same function as Reorder but in a constant O(n) way.
   * <!-- TODO(zeshunzong): Technically we can reduce the number of copies
   * introducing a flag and alternatingly return the (currently) ordered
   * attribute. Since we haven't decided which algorithm to use, for clarity
   * let's defer this for future. -->
   * <!-- TODO(zeshunzong): May need to reorder more attributes as more
   * attributes are added. -->
   */
  void Reorder2(const std::vector<size_t>& new_order);

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
  const std::vector<Matrix3<T>>& trial_deformation_gradients() const {
    return trial_deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& elastic_deformation_gradients() const {
    return elastic_deformation_gradients_;
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
   * Returns the trial deformation gradient of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetTrialDeformationGradientAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return trial_deformation_gradients_[p];
  }

  /**
   * Returns the elastic deformation gradient of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetElasticDeformationGradientAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles_);
    return elastic_deformation_gradients_[p];
  }

  /**
   * Returns the B_matrix of p-th particle. B_matrix is part of the affine
   * momentum matrix C as
   * v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
   * See (173) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
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
   * Sets the trial deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetTrialDeformationGradient(size_t p, const Matrix3<T>& F_trial_in) {
    DRAKE_ASSERT(p < num_particles_);
    trial_deformation_gradients_[p] = F_trial_in;
  }

  /**
   * Sets the elastic deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetElasticDeformationGradient(size_t p, const Matrix3<T>& FE_in) {
    DRAKE_ASSERT(p < num_particles_);
    elastic_deformation_gradients_[p] = FE_in;
  }

  /**
   * Sets the B_matrix at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetBMatrix(size_t p, const Matrix3<T>& B_matrix) {
    DRAKE_ASSERT(p < num_particles_);
    B_matrices_[p] = B_matrix;
  }

  // TODO: do whatever needed to compute values in 
  void UpdatePadSplatters(){

  }

 private:
  size_t num_particles_ = 0;
  std::vector<Vector3<T>> positions_{};
  std::vector<Vector3<T>> velocities_{};
  std::vector<T> masses_{};
  std::vector<T> reference_volumes_{};

  std::vector<Matrix3<T>> trial_deformation_gradients_{};
  std::vector<Matrix3<T>> elastic_deformation_gradients_{};

  // The affine matrix B_p in APIC
  // B_matrix is part of the affine momentum matrix C as
  // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
  std::vector<Matrix3<T>> B_matrices_{};

  // TODO(zeshunzong): Consider make struct Scratch and put the buffer data
  // inside scratch for better clarity. for reorder only
  std::vector<T> temporary_scalar_field_{};
  std::vector<Vector3<T>> temporary_vector_field_{};
  std::vector<Matrix3<T>> temporary_matrix_field_{};

  std::vector<PadSplatter<T>> pad_splatters_{}; // store weight and dw
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
