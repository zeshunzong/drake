#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmModel {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmModel);

  MpmModel() {}

  void SetUpTransferAndUpdateCurrentGridData(Particles<T>* particles,
                                             GridData<T>* grid_data,
                                             SparseGrid<T>* sparse_grid) {
    transfer_.SetUpTransfer(sparse_grid, particles);
    transfer_.P2G(*particles, *sparse_grid, grid_data);
  }

  void ComputeDeformationScratch(const GridData<T>& grid_data,
                                 const SparseGrid<T>& sparse_grid,
                                 const Particles<T>& particles, double dt) {
    ParticlesData<T> particles_data{};  // TODO(zeshunzong): attribute?
    transfer_.G2P(sparse_grid, grid_data, particles, &particles_data);
    particles.ComputeDeformationScratch(particles_data.particle_grad_v_next, dt,
                                        &scratch_);
  }

  /**
   * Computes the elastic energy due to deformation.
   * @pre scratch_ has been computed from grid velocity.
   */
  T ComputeElasticEnergy(const Particles<T>& particles) const {
    return particles.ComputeTotalElasticEnergy(
        scratch_.elastic_deformation_gradients);
  }

  /**
   * Kinetic energy + gravity
   * TODO(zeshunzong): write formulas here
   */
  T ComputeKineticAndGravitationalEnergy(const GridData<T>& current_grid,
                                         const GridData<T>& new_grid,
                                         double dt) const {
    DRAKE_ASSERT(current_grid.num_active_nodes() ==
                 new_grid.num_active_nodes());
    T sum = 0;

    for (size_t i = 0; i < current_grid.num_active_nodes(); ++i) {
      const Vector3<T>& dv =
          current_grid.GetVelocityAt(i) - new_grid.GetVelocityAt(i);
      sum += 0.5 * current_grid.GetMassAt(i) * (dv).squaredNorm();
      sum += dt * current_grid.GetMassAt(i) * dv.dot(gravity_);
    }
    return sum;
  }

  /**
   * Total energy = elastic energy + kinetic energy + gravitational energy
   * @pre scratch_ has been computed from grid velocity.
   */
  T ComputeTotalEnergy(const Particles<T>& particles,
                       const GridData<T>& current_grid,
                       const GridData<T>& new_grid, double dt) const {
    return ComputeElasticEnergy(particles) +
           ComputeKineticAndGravitationalEnergy(current_grid, new_grid, dt);
  }

  /**
   * Computes the elastic force due to elastic deformation. It implicitly
   * depends on current_grid (grid velocites) through scratch_.PK_stresses.
   * @pre scratch_ has been computed from grid velocity.
   */
  void ComputeElasticForce(const Particles<T>& particles,
                           const SparseGrid<T>& sparse_grid,
                           std::vector<Vector3<T>>* grid_forces) {
    transfer_.ComputeGridElasticForces(particles, sparse_grid,
                                       scratch_.PK_stresses, grid_forces);
  }

  void AddKineticAndGravitationalForce(
      const GridData<T>& current_grid, const GridData<T>& new_grid, double dt,
      std::vector<Vector3<T>>* grid_forces) const {
    std::vector<Vector3<T>>& grid_forces_ref = *grid_forces;
    for (size_t i = 0; i < current_grid.num_active_nodes(); ++i) {
      const Vector3<T>& dv =
          current_grid.GetVelocityAt(i) - new_grid.GetVelocityAt(i);
      grid_forces_ref[i] += current_grid.GetMassAt(i) * dv / dt;
      grid_forces_ref[i] += current_grid.GetMassAt(i) * gravity_;
    }
  }

  /**
   * @pre scratch_ has been computed from grid velocity.
   */
  void ComputeElasticHessian(const Particles<T>& particles,
                             const SparseGrid<T>& sparse_grid,
                             MatrixX<T>* hessian) const {
    transfer_.ComputeGridElasticHessian(
        particles, sparse_grid, scratch_.dPdF_contractF0_contractF0, hessian);
  }

  void ComputeElasticHessianTimesZ(
      const Particles<T>& particles, const SparseGrid<T>& sparse_grid,
      const std::vector<Vector3<T>>& z,
      std::vector<Vector3<T>>* hessian_times_z) const {
    transfer_.ComputeElasticHessianTimesZ(z, particles, sparse_grid,
                                          scratch_.dPdF, hessian_times_z);
  }

 private:
  Vector3<T> gravity_{0.0, 0.0, -9.8};
  MpmTransfer<T> transfer_;

  DeformationScratch<T> scratch_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
