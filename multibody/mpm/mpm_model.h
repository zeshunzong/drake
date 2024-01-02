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
                                        // F, P, dPdF
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

 private:
  Vector3<T> gravity_{0.0, 0.0, -9.8};
  MpmTransfer<T> transfer_;

  DeformationScratch<T> scratch_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
