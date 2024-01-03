#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class DeformationState {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DeformationState);

  DeformationState(size_t num_particles) {
    Fs_.resize(num_particles);
    Ps_.resize(num_particles);
    dPdFs_.resize(num_particles);
  }

  /**
   * Computes F, P and dPdF for each particle, from the grid_data
   * @pre !particles.NeedReordering()
   * @pre sparse_grid is compatible with current particles
   */
  void Update(const GridData<T>& grid_data, const SparseGrid<T>& sparse_grid,
              const Particles<T>& particles, const MpmTransfer<T>& transfer,
              double dt, TransferScratch<T>* scratch) {
    ParticlesData<T> particles_data{};  // TODO(zeshunzong): attribute?
    transfer.G2P(sparse_grid, grid_data, particles, &particles_data, scratch);
    particles.ComputeFsPsdPdFs(particles_data.particle_grad_v_next, dt, &Fs_,
                               &Ps_, &dPdFs_);
  }

  const std::vector<Matrix3<T>>& Fs() const { return Fs_; }
  const std::vector<Matrix3<T>>& Ps() const { return Ps_; }
  const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs() const { return dPdFs_; }

 private:
  std::vector<Matrix3<T>> Fs_{};
  std::vector<Matrix3<T>> Ps_{};
  std::vector<Eigen::Matrix<T, 9, 9>> dPdFs_{};
};

template <typename T>
class MpmModel {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmModel);

  MpmModel() {}

  T ComputeElasticEnergy(const Particles<T>& particles,
                         const DeformationState<T>& deformation_state) const {
    return particles.ComputeTotalElasticEnergy(deformation_state.Fs());
  }

  /**
   * Computes the elastic force due to elastic deformation.
   */
  void ComputeElasticForce(const Particles<T>& particles,
                           const SparseGrid<T>& sparse_grid,
                           const MpmTransfer<T>& transfer,
                           const DeformationState<T>& deformation_state,
                           std::vector<Vector3<T>>* grid_forces,
                           TransferScratch<T>* scratch) const {
    transfer.ComputeGridElasticForces(
        particles, sparse_grid, deformation_state.Ps(), grid_forces, scratch);
  }

 private:
  Vector3<T> gravity_{0.0, 0.0, -9.81};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
