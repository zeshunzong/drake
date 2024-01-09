#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "drake/multibody/mpm/matrix_replacement.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmDriver {
 public:
  MpmDriver(double h, double dt) : sparse_grid_(h) {
    DRAKE_DEMAND(dt > 0);
    dt_ = dt;
  }

  void AddParticle(
      const Vector3<T>& position, const Vector3<T>& velocity,
      std::unique_ptr<mpm::constitutive_model::ElastoPlasticModel<T>>
          constitutive_model,
      const T& mass, const T& reference_volume) {
    particles_.AddParticle(position, velocity, std::move(constitutive_model),
                           mass, reference_volume);
  }

  int AdvanceDt() {
    // TODO(zeshunzong): precondition
    // TODO(zeshunzong): line search
    // TODO(zeshunzong): projection
    // TODO(zeshunzong): test matrix free or not
    int num_newtons = ComputeGridVelocities();
    std::cout << "num newtons: " << num_newtons << std::endl;
    // now we have G
    // SAP can take G and modify it if needed
    UpdateParticlesFromGridData();

    return num_newtons;
  }

  int ComputeGridVelocities() {
    transfer_.SetUpTransfer(&sparse_grid_, &particles_);
    transfer_.P2G(particles_, sparse_grid_, &grid_data_, &scratch_);
    DeformationState<T> deformation_state(particles_, sparse_grid_, grid_data_);
   
    v_prev_ = grid_data_.velocities();

    int count = 0;
    dG_norm_ = std::numeric_limits<T>::infinity();
    while (dG_norm_ > newton_epsilon_) {
      if (count > max_newton_iter_) {
        break;
      }
      deformation_state.Update(transfer_, dt_, &scratch_);
      // find minus_gradient
      model_.ComputeMinusDEnergyDV(transfer_, v_prev_, deformation_state, dt_,
                                   &minus_dEdv_, &scratch_);

      // find dG_ = hessian^-1 * minus_gradient
      if (matrix_free_) {
        MatrixReplacement<T> hessian_operator =
            MatrixReplacement<T>(model_, deformation_state, transfer_, dt_);
        cg_matrix_free_.compute(hessian_operator);
        dG_ = cg_matrix_free_.solve(minus_dEdv_);

      } else {
        model_.ComputeD2EnergyDV2(transfer_, deformation_state, dt_, &d2Edv2_);

        cg_dense_.compute(d2Edv2_);
        dG_ = cg_dense_.solve(minus_dEdv_);
      }

      grid_data_.AddDG(dG_);
      dG_norm_ = dG_.norm();

      ++count;
    }
    return count;
  }

  void UpdateParticlesFromGridData() {
    transfer_.G2P(sparse_grid_, grid_data_, particles_, &particles_data_,
                  &scratch_);

    // update F_trial, F_elastic, stress, B_matrix
    transfer_.UpdateParticlesState(particles_data_, dt_, &particles_);
    // update particle position, this is the last step
    particles_.AdvectParticles(dt_);
  }

  const Particles<T>& particles() const { return particles_; }
  const MpmModel<T>& mpm_model() const { return model_; }

  void SetMatrixFree(bool matrix_free) {
    matrix_free_ = matrix_free;
  }

 private:
  double dt_ = 0.0;

  Particles<T> particles_;
  SparseGrid<T> sparse_grid_;
  GridData<T> grid_data_;
  std::vector<Vector3<T>> v_prev_;
  ParticlesData<T> particles_data_;

  MpmModel<T> model_;

  MpmTransfer<T> transfer_;
  TransferScratch<T> scratch_;

  double newton_epsilon_ = 0.0001;

  Eigen::VectorX<T> dG_;
  T dG_norm_ = std::numeric_limits<T>::infinity();

  Eigen::VectorX<T> minus_dEdv_;
  MatrixX<T> d2Edv2_;

  bool matrix_free_ = false;

  Eigen::ConjugateGradient<MatrixReplacement<T>, Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
      cg_matrix_free_;
  Eigen::ConjugateGradient<MatrixX<T>, Eigen::Lower | Eigen::Upper> cg_dense_;

  int max_newton_iter_ = 500;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
