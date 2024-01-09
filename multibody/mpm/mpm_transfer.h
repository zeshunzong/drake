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

template <typename T>
struct TransferScratch {
  // scratch pads for transferring states from particles to grid nodes
  // there will be one pad for each particle
  std::vector<P2gPad<T>> p2g_pads{};
  // scratch pad for transferring states from grid nodes to particles
  G2pPad<T> g2p_pad{};
};

/**
 * An implementation of MPM's transfer schemes. We follow Section 10.5 in
 * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
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
           GridData<T>* grid_data, TransferScratch<T>* scratch) const;

  /**
   * Grid to particles transfer.
   * See Section 10.1 and 10.2 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given grid_data, writes particle v, B, and grad_v to particles_data.
   */
  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data,
           const Particles<T>& particles, ParticlesData<T>* particles_data,
           TransferScratch<T>* scratch) const;

  /**
   * Updates velocity, B-matrix, trial deformation gradient, elastic deformation
   * gradient, and stress P for all particles.
   * @note Return mapping and constitutive model are invoked to get elastic
   * deformation gradient and stress.
   * @note This updates everything except for particle positions.
   */
  void UpdateParticlesState(const ParticlesData<T>& particles_data, double dt,
                            Particles<T>* particles) const;

  /**
   * Computes the grid_forces at all active grid nodes.
   * fᵢ = − ∑ₚ V⁰ₚ P F₀ᵀ ∇ wᵢₚ
   * Also see the accompanied energy_derivatives.md.
   * @note the computation is similar to P2G, with only grid forces being
   * computed.
   * @note the computation depends implicitly on the grid velocities through
   * the first Piola-Kirchhoff stresses `Ps`, which should be computed from the
   * grid velocities.
   * @note the computation depends on the current elastic_deformation_gradient
   * F₀ stored in particles (for chain rule).
   * @pre Ps.size() == particles.num_particles().
   * @note fᵢₐ = grid_elastic_forces(3*i+a).
   */
  void ComputeGridElasticForces(const Particles<T>& particles,
                                const SparseGrid<T>& grid,
                                const std::vector<Matrix3<T>>& Ps,
                                Eigen::VectorX<T>* grid_elastic_forces,
                                TransferScratch<T>* scratch) const {
    particles.SplatStressToP2gPads(Ps, &(scratch->p2g_pads));
    grid.GatherForceFromP2gPads(scratch->p2g_pads, grid_elastic_forces);
  }

  /**
   * Computes the second order derivative of elastic energy w.r.t. grid
   * positions. See the accompanied energy_derivatives.md.
   */
  void ComputeGridElasticHessian(
      const Particles<T>& particles, const SparseGrid<T>& grid,
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs,
      MatrixX<T>* hessian) const;

  /**
   * Computes result += d2(elastic_energy)/dv2 * z.
   * @note d2(elastic_energy)/dv2 = ComputeGridElasticHessian() * dt * dt.
   */
  void AddD2ElasticEnergyDV2TimesZ(
      const Eigen::VectorX<T>& z, const Particles<T>& particles,
      const SparseGrid<T>& grid,
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs, double dt,
      Eigen::VectorX<T>* result) const;

 private:
  // Adds the batch_index_3d-th pad hessian (of elastic energy w.r.t grid
  // positions) to the hessian.
  void AddPadHessianToHessian(const Vector3<int> batch_index_3d,
                              const SparseGrid<T>& grid,
                              const MatrixX<T>& pad_hessian,
                              MatrixX<T>* hessian) const;

  // Computes the matrix A in the computation of ElasticHessianTimesZ, for
  // particle p. See energy_derivatives.md for details.
  void ComputeAp(size_t p, const Eigen::VectorX<T>& z,
                 const Particles<T>& particles, const SparseGrid<T>& grid,
                 const Eigen::Matrix<T, 9, 9>& dPdF, Matrix3<T>* Ap) const;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake