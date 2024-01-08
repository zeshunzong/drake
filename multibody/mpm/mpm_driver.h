#pragma once

#include <array>
#include <iostream>
#include <memory>
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
    transfer_.SetUpTransfer(&sparse_grid_, &particles_);
    transfer_.P2G(particles_, sparse_grid_, &grid_data_, &scratch_);

    DeformationState<T> deformation_state(particles_, sparse_grid_, grid_data_);
    v_prev_ = grid_data_.velocities();
    dG_.resize(sparse_grid_.num_active_nodes() * 3);  // optional?

    int count = 0;

    while (dG_norm_ > newton_epsilon_) {
      if (count > max_newton_iter_) {
        break;
      }
      deformation_state.Update(transfer_, dt_, &scratch_);

      // find minus_gradient
      model_.ComputeMinusDEnergyDV(transfer_, v_prev_, deformation_state, dt_,
                                   &minus_dEdv_, &scratch_);

      Eigen::VectorX<T> minus_gradient(grid_data_.num_active_nodes() * 3);
      for (size_t i = 0; i < grid_data_.num_active_nodes(); ++i) {
        minus_gradient.segment(3 * i, 3) = minus_dEdv_[i];
      }

      // find hessian^-1 * minus_gradient
      if (matrix_free_) {
        MatrixReplacement<T> hessian_operator =
            MatrixReplacement<T>(model_, deformation_state, transfer_, dt_);
        cg_matrix_free_.compute(hessian_operator);
        dG_ = cg_matrix_free_.solve(minus_gradient);
        std::cout << "#iterations:     " << cg_matrix_free_.iterations() <<
        std::endl; std::cout << "estimated error: " <<
        cg_matrix_free_.error() << std::endl;
      } else {
        model_.ComputeD2EnergyDV2(transfer_, deformation_state, dt_, &d2Edv2_);
        cg_dense_.compute(d2Edv2_);
        dG_ = cg_dense_.solve(minus_gradient);
        // std::cout << "#iterations:     " << cg_dense_.iterations() <<
        // std::endl; std::cout << "estimated error: " << cg_dense_.error() <<
        // std::endl;
      }

      grid_data_.AddDG(dG_);
      dG_norm_ = dG_.norm();

      ++count;
    }

    // now we have dG, give it back to particles
    ParticlesData<T> particles_data;
    transfer_.G2P(sparse_grid_, grid_data_, particles_, &particles_data,
                  &scratch_);

    std::cout << particles_data.particle_velocites_next[0](0) << " "
              << particles_data.particle_velocites_next[0](1) << " "
              << particles_data.particle_velocites_next[0](2) << " "
              << std::endl;
    return count;

    // TODO: hessian times Z for wrapper rewrite for eigenXd
    // TODO: compute gradient for eigenXd
    // TODO: multiple steps
    // TODO: precondition
    // TODO: line search
    // TODO: projection
    // TODO: test matrix free or not
    // TODO: set gravity
  }

 private:
  double dt_ = 0.0;

  Particles<T> particles_;
  SparseGrid<T> sparse_grid_;
  GridData<T> grid_data_;
  std::vector<Vector3<T>> v_prev_;

  MpmModel<T> model_;

  MpmTransfer<T> transfer_;
  TransferScratch<T> scratch_;

  double newton_epsilon_ = 0.0001;

  Eigen::VectorX<T> dG_;
  T dG_norm_ = std::numeric_limits<T>::infinity();

  std::vector<Vector3<T>> minus_dEdv_;
  MatrixX<T> d2Edv2_;

  bool matrix_free_ = true;

  Eigen::ConjugateGradient<MatrixReplacement<T>, Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
      cg_matrix_free_;
  Eigen::ConjugateGradient<MatrixX<T>, Eigen::Lower | Eigen::Upper> cg_dense_;

  int max_newton_iter_ = 500;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
