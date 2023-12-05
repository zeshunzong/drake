#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/pad.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/particles_data.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * An implementation of MPM's transfer schemes. We follow Section 10.5 in
 * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
 *
 * Have the following things in hand to advance from tₙ to tₙ₊₁:
 *
 * Particles<double> particles; -- carries state of Lagrangian particles at tₙ
 * SparseGrid<double> sparse_grid; -- auxiliary structure for indexing
 * GridData<double> grid_data; -- zero-initialized, to hold grid state
 * MpmTransfer<double> mpm_transfer; -- transfer functions
 * ParticlesData<double> particles_data; -- data holder for particles
 *
 * @note Below Gₙ represents grid velocities, Pₙ represents all states stored
 * on particles.
 *
 * In: Pₙ particles at tₙ
 * Out: Gₙ₊₁ grid state at tₙ₊₁, Pₙ₊₁ particles at tₙ₊₁.
 *
 * Explicit temporal advancement:
 * ```
 * mpm_transfer.SetupTransfer(&sparse_grid, &particles);
 * // Gₙ = P2G(Pₙ)
 * mpm_transfer.P2G(particles, grid, &grid_data);
 * // compute Gₙ₊₁
 * TODO(zeshunzong): mpm_transfer.AddGravity(&grid_data);
 * // particle_data = G2P(Gₙ₊₁)
 * mpm_transfer.G2P(grid, grid_data, particles, &particles_data);
 * // update particles state except for position
 * mpm_transfer.UpdateParticlesState(particles_data, dt, &particles);
 * // afterwards we get Pₙ₊₁
 * particles.AdvectParticles(dt);
 * ```
 *
 * Implicit temporal advancement:
 * ```
 * mpm_transfer.SetupTransfer(&sparse_grid, &particles);
 * dG = ∞
 * while |dG|>ε:
 *   // Gₙ = P2G(Pₙ)
 *   mpm_transfer.P2G(particles, grid, &grid_data);
 *   // Solve the linear system H(Pₙ)⋅dG = -f(Pₙ)
 *   @note H and f depend on weights and particle deformation gradients
 *   TODO(zeshunzong): LinearSolve(H, f, Pₙ, &dg)
 *   Gₙ = Gₙ + dG
 *   // particle_data = G2P(Gₙ₊₁)
 *   mpm_transfer.G2P(grid, grid_data, particles, &particles_data);
 *   // Pₙ = UpdateParticlesState(particle_data)
 *   mpm_transfer.UpdateParticlesState(particles_data, dt, &particles);
 * end
 * // afterwards we get Pₙ₊₁
 * particles.AdvectParticles(dt);
 * Output Gₙ₊₁:=Gₙ, Pₙ₊₁:=Pₙ.
 * ```
 */
template <typename T>
class MpmTransfer {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmTransfer);

  MpmTransfer() {}

  /**
   * Given current configuration of grid and particles, performs all necessary
   * preparations for transferring, including sorting and computing weights.
   */
  void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles) const;

  /**
   * Particles to grid transfer.
   * See Section 10.1 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given particle data, writes to grid_data.
   * @note grid_data is first cleared before "transferring" particle data to
   * grid_data.
   */
  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* grid_data);

  /**
   * Grid to particles transfer.
   * See Section 10.1 and 10.2 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given grid_data, writes particle v, B, and grad_v to particles_data.
   */
  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data,
           const Particles<T>& particles, ParticlesData<T>* particles_data);

  /**
   * Updates velocity, B-matrix, trial deformation gradient, elastic deformation
   * gradient, and stress τ for all particles.
   * @note Return mapping and constitutive model are invoked to get elastic
   * deformation gradient and stress.
   * @note This updates everything except for particle positions.
   * TODO(zeshunzong): finish return mapping and stress computation
   */
  void UpdateParticlesState(const ParticlesData<T>& particles_data, double dt,
                            Particles<T>* particles) const;

 private:
  // scratch pads for transferring states from particles to grid nodes
  std::vector<P2gPad<T>> p2g_pads_{};

  // scratch pad for transferring states from grid nodes to particles
  G2pPad<T> g2p_pad_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
