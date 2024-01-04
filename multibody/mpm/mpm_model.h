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
 *    state.Update(mpm_transfer, dt, scratch);
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

  DeformationState(Particles<T>& particles, const SparseGrid<T>& sparse_grid,
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

  T ComputeElasticEnergy(const DeformationState<T>& deformation_state) const {
    return deformation_state.particles().ComputeTotalElasticEnergy(
        deformation_state.Fs());
  }

  /**
   * Computes the elastic force due to elastic deformation.
   */
  void ComputeElasticForce(const MpmTransfer<T>& transfer,
                           const DeformationState<T>& deformation_state,
                           std::vector<Vector3<T>>* grid_forces,
                           TransferScratch<T>* scratch) const {
    transfer.ComputeGridElasticForces(
        deformation_state.particles(), deformation_state.sparse_grid(),
        deformation_state.Ps(), grid_forces, scratch);
  }

  /**
   * Computes the minus of the derivative force w.r.t. grid positions.
   */
  void ComputeElasticHessian(const MpmTransfer<T>& transfer,
                             const DeformationState<T>& deformation_state,
                             MatrixX<T>* hessian) const {
    transfer.ComputeGridElasticHessian(deformation_state.particles(),
                                       deformation_state.sparse_grid(),
                                       deformation_state.dPdFs(), hessian);
  }

  void ComputeElasticHessianTimesZ(
      const std::vector<Vector3<T>>& z, const MpmTransfer<T>& transfer,
      const DeformationState<T>& deformation_state,
      std::vector<Vector3<T>>* hessian_times_z) const {
    transfer.ComputeGridElasticHessianTimesZ(
        z, deformation_state.particles(), deformation_state.sparse_grid(),
        deformation_state.dPdFs(), hessian_times_z);
  }

 private:
  Vector3<T> gravity_{0.0, 0.0, -9.81};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
