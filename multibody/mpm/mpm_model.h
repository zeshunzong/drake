#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * Advance from tₙ to tₙ₊₁ using implicit scheme:
 *
 * Have the following objects in hands
 * Particles<double> particles; -- carries state of Lagrangian particles at tₙ,
 * this is the only meaningful data at this moment
 *
 * SparseGrid<double> sparse_grid; -- auxiliary structure for indexing
 *
 * GridData<double> grid_data; -- zero-initialized
 * ParticlesData<double> particles_data; -- zero-initialized
 *
 * MpmModel<double> mpm_model; -- compute energy and its derivatives
 *
 * MpmTransfer<double> mpm_transfer; -- transfer functions
 * TransferScratch<double> scratch; -- scratch data for transfer
 *
 * dG = ∞
 * mpm_transfer.SetUpTransfer(&sparse_grid, &particles);
 * mpm_transfer.P2G(particles, sparse_grid, &grid_data, &transfer_scratch);
 *
 * DeformationState<double> state(particles, sparse_grid, grid_data);
 * while |dG|>ε:
 *    state.Update(mpm_transfer, dt, &scratch);
 *    energy = mpm_model.ComputeEnergy(state);
 *    force = mpm_model.ComputeForce(state, mpm_transfer, scratch);
 *    hessian = mpm_model.ComputeHessian(state, mpm_transfer);
 *    dG = LinearSolve(force, energy);
 *    grid_data += dG
 *
 * particles at tₙ₊₁ = mpm_transfer.G2P(grid_data, dt) and
 * mpm_transfer.UpdateParticlesState(), updating v, C, F, P, x, to be stated
 * more clearly
 *
 * TODO(zeshunzong): write an explicit scheme
 */

template <typename T>
class DeformationState {
  /**
   * A temporary holder for data needed in Newton iterations.
   */
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformationState);
  // NOLINTNEXTLINE
  DeformationState(Particles<T>& particles, const SparseGrid<T>& sparse_grid,
                   // NOLINTNEXTLINE
                   GridData<T>& grid_data)
      : particles_(particles),
        sparse_grid_(sparse_grid),
        grid_data_(grid_data) {
    Fs_.resize(particles.num_particles());
    Ps_.resize(particles.num_particles());
    dPdFs_.resize(particles.num_particles());
  }

  /**
   * Computes F, P and dPdF for each particle, from the grid_data
   * @pre !particles.NeedReordering()
   * @pre sparse_grid is compatible with current particles
   */
  void Update(const MpmTransfer<T>& transfer, double dt,
              TransferScratch<T>* scratch) {
    ParticlesData<T> particles_data{};  // TODO(zeshunzong): attribute?
    // TODO(zeshunzong): some other data is computed but not used in
    // particles_data
    transfer.G2P(sparse_grid_, grid_data_, particles_, &particles_data,
                 scratch);
    particles_.ComputeFsPsdPdFs(particles_data.particle_grad_v_next, dt, &Fs_,
                                &Ps_, &dPdFs_);
  }

  const std::vector<Matrix3<T>>& Fs() const { return Fs_; }
  const std::vector<Matrix3<T>>& Ps() const { return Ps_; }
  const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs() const { return dPdFs_; }

  const Particles<T>& particles() const { return particles_; }
  const SparseGrid<T>& sparse_grid() const { return sparse_grid_; }
  const GridData<T>& grid_data() const { return grid_data_; }

 private:
  std::vector<Matrix3<T>> Fs_{};
  std::vector<Matrix3<T>> Ps_{};
  std::vector<Eigen::Matrix<T, 9, 9>> dPdFs_{};

  Particles<T>& particles_;
  const SparseGrid<T>& sparse_grid_;
  GridData<T>& grid_data_;
};

template <typename T>
class MpmModel {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmModel);

  MpmModel() {}

  /**
   * Total energy = elastic energy + kinetic energy + gravitational energy.
   */
  T ComputeEnergy(const std::vector<Vector3<T>>& v_prev,
                  const DeformationState<T>& deformation_state,
                  double dt) const {
    return ComputeElasticEnergy(deformation_state) +
           ComputeKineticAndGravitationalEnergy(v_prev, deformation_state, dt);
  }

  /**
   * elastic energy = ∑ₚ V⁰ₚ ψ(Fₚ), where Fₚ depends on grid velocities
   */
  T ComputeElasticEnergy(const DeformationState<T>& deformation_state) const {
    return deformation_state.particles().ComputeTotalElasticEnergy(
        deformation_state.Fs());
  }

  /**
   * Computes - d(total_energy)/dv = -d(elastic_energy)/dv -
   * d(kinetic_energy)/dv - d(gravitational_energy)/dv.
   */
  void ComputeMinusDEnergyDV(const MpmTransfer<T>& transfer,
                             const std::vector<Vector3<T>>& v_prev,
                             const DeformationState<T>& deformation_state,
                             double dt, Eigen::VectorX<T>* minus_dedv,
                             TransferScratch<T>* scratch) const;

  /**
   * Computes -d(elastic_energy)/dv.
   */
  void ComputeMinusDElasticEnergyDV(
      const MpmTransfer<T>& transfer,
      const DeformationState<T>& deformation_state, double dt,
      Eigen::VectorX<T>* dEnergydV, TransferScratch<T>* scratch) const;

  /**
   * Computes the 3*num_active_nodes()by3*num_active_nodes() hessian matrix d^2
   * (total_energy) / dv^2.
   */
  void ComputeD2EnergyDV2(const MpmTransfer<T>& transfer,
                          const DeformationState<T>& deformation_state,
                          double dt, MatrixX<T>* hessian) const;

  /**
   * Computes the 3*num_active_nodes()by3*num_active_nodes() hessian matrix d^2
   * (elastic_energy) / dv^2.
   */
  void ComputeD2ElasticEnergyDV2(const MpmTransfer<T>& transfer,
                                 const DeformationState<T>& deformation_state,
                                 double dt, MatrixX<T>* hessian) const;

  /**
   * Computes result += ComputeD2EnergyDV2() * z.
   */
  void AddD2EnergyDV2TimesZ(const Eigen::VectorX<T>& z,
                            const MpmTransfer<T>& transfer,
                            const DeformationState<T>& deformation_state,
                            double dt, Eigen::VectorX<T>* result) const;

  /**
   * Computes result += ComputeD2ElasticEnergyDV2() * z.
   */
  void AddD2ElasticEnergyDV2TimesZ(const Eigen::VectorX<T>& z,
                                   const MpmTransfer<T>& transfer,
                                   const DeformationState<T>& deformation_state,
                                   double dt, Eigen::VectorX<T>* result) const;

  void SetGravity(const Vector3<T>& g) { gravity_ = g; }

  const Vector3<T>& gravity() const { return gravity_; }

  // precondition for cg solve, currently this is MASS
  // that is, in absense of elastic deformation, this is equivalent to diagonal
  // preconditioner
  void Precondition(const DeformationState<T>& deformation_state,
                    const Eigen::VectorX<T>& rhs, Eigen::VectorX<T>* x) const {
    for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
         ++i) {
      if (deformation_state.grid_data().masses()[i] > 0.0) {
        (*x).segment(3 * i, 3) =
            rhs.segment(3 * i, 3) / deformation_state.grid_data().masses()[i];
      } else {
        (*x).segment(3 * i, 3) = rhs.segment(3 * i, 3);
      }
    }
  }

 private:
  // Kinetic energy = 0.5 * m * (v - v_prev)ᵀ(v - v_prev).
  // Gravitational energy = - m*dt*gᵀv. Since we only care about its gradient,
  // we can do - m*dt*gᵀ(v - v_prev).
  T ComputeKineticAndGravitationalEnergy(
      const std::vector<Vector3<T>>& v_prev,
      const DeformationState<T>& deformation_state, double dt) const;

  // -d(kinetic_energy)/dv = - m * (v - v₀).
  // -d(gravitational_energy)/dv = dt * mg.
  // result = result -d(kinetic_energy)/dv -d(gravitational_energy)/dv
  void MinusDKineticEnergyDVAndDGravitationalEnergyDV(
      const std::vector<Vector3<T>>& v_prev,
      const DeformationState<T>& deformation_state, double dt,
      Eigen::VectorX<T>* result) const;

  Vector3<T> gravity_{0.0, 0.0, -10.0};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
