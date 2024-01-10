#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "drake/math/rigid_transform.h"
#include "drake/multibody/mpm/internal/analytic_level_set.h"
#include "drake/multibody/mpm/internal/poisson_disk.h"
#include "drake/multibody/mpm/matrix_replacement.h"
#include "drake/multibody/mpm/particles_to_bgeo.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmDriver {
 public:
  MpmDriver(double h, double dt) : sparse_grid_(h) {
    DRAKE_DEMAND(h > 0);
    DRAKE_DEMAND(dt > 0);
    h_ = h;
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

  void AddParticlesViaPoissonDiskSampling(
      const internal::AnalyticLevelSet& level_set,
      const math::RigidTransform<double>& pose,
      const mpm::constitutive_model::ElastoPlasticModel<T>& elastoplastic_model) {
    const std::array<Vector3<double>, 2> bounding_box =
        level_set.bounding_box();
    double min_num_particles_per_cell = 1;
    double sample_r = h_ / (std::cbrt(min_num_particles_per_cell) + 1);

    std::array<double, 3> xmin = {bounding_box[0][0], bounding_box[0][1],
                                  bounding_box[0][2]};
    std::array<double, 3> xmax = {bounding_box[1][0], bounding_box[1][1],
                                  bounding_box[1][2]};
    // Generate sample particles in the reference frame
    std::vector<Vector3<double>> particles_sample_positions =
        internal::PoissonDiskSampling(sample_r, xmin, xmax);

    // Pick out sampled particles that are in the object
    int num_samples = particles_sample_positions.size();
    std::vector<Vector3<double>> particles_positions, particles_velocities;
    for (int p = 0; p < num_samples; ++p) {
      // Denote the particle by P, the object by B, the frame centered at the
      // origin of the object by Bo, and the frame of the particle by Bp.

      // The pose and spatial velocity of the object in the world frame
      const math::RigidTransform<double>& X_WB = pose;

      // Sample particles and get the position of the particle with respect to
      // the object in the object's frame
      const Vector3<double>& p_BoBp_B = particles_sample_positions[p];

      // If the particle is in the level set
      if (level_set.IsInClosure(p_BoBp_B)) {
        // Place the particle's position in world frame
        particles_positions.emplace_back(X_WB * p_BoBp_B);
      }
    }

    int num_particles = particles_positions.size();
    // We assume every particle have the same volume and mass
    double reference_volume_p = level_set.volume() / num_particles;
    double mass_p = 1000.0 * reference_volume_p;

    for (size_t p = 0; p < particles_positions.size(); ++p) {
      AddParticle(particles_positions[p], Vector3<T>(0, 0, 0),
                  elastoplastic_model.Clone(),
                      mass_p,
                  reference_volume_p);
    }
  }

  int AdvanceDt() {
    // TODO(zeshunzong): precondition
    // TODO(zeshunzong): line search

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

    // if (apply_ground_) {
    //   // identify grid nodes that penetrates ground
    //   UpdateCollisionNodesWithGround();
    // }

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

      // if (apply_ground_) {
      //   ProjectCollisionGround(&minus_dEdv_);
      // }

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

      // if (apply_ground_) {
      //   ProjectCollisionGround(&dG_);
      // }

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

    for (size_t i = 0; i < particles_.num_particles(); ++i) {
      std::cout << particles_.velocities()[i](2) << std::endl;
    }
  }

  const Particles<T>& particles() const { return particles_; }
  const MpmModel<T>& mpm_model() const { return model_; }

  void SetMatrixFree(bool matrix_free) { matrix_free_ = matrix_free; }

  void WriteParticlesToBgeo(int io_step) {
    std::string output_filename = "./f" + std::to_string(io_step) + ".bgeo";
    internal::WriteParticlesToBgeo(output_filename, particles_.positions(),
                                   particles_.velocities(),
                                   particles_.masses());
  }

 private:
  // TODO(zeshunzong): only sticky ground right now
  void UpdateCollisionNodesWithGround() {
    collision_nodes_.clear();
    for (size_t i = 0; i < sparse_grid_.num_active_nodes(); ++i) {
      if (sparse_grid_.To3DIndex(i)(2) <= 0) {
        collision_nodes_.push_back(i);
      }
    }
  }

  void ProjectCollisionGround(Eigen::VectorX<T>* v) const {
    for (auto node_idx : collision_nodes_) {
      if (sticky_ground_) {
        (*v).segment(3 * node_idx, 3) = Vector3<T>(0, 0, 0);
      } else {
        (*v)(3 * node_idx + 2) = 0.0;
      }
    }
  }

  double h_ = 0.0;
  double dt_ = 0.0;

  Particles<T> particles_;
  SparseGrid<T> sparse_grid_;
  GridData<T> grid_data_;
  std::vector<Vector3<T>> v_prev_;
  ParticlesData<T> particles_data_;

  std::vector<size_t> collision_nodes_;

  MpmModel<T> model_;

  MpmTransfer<T> transfer_;
  TransferScratch<T> scratch_;

  double newton_epsilon_ = 1e-6;

  Eigen::VectorX<T> dG_;
  T dG_norm_ = std::numeric_limits<T>::infinity();

  Eigen::VectorX<T> minus_dEdv_;
  MatrixX<T> d2Edv2_;

  bool matrix_free_ = false;

  bool sticky_ground_ = true;

  bool apply_ground_ = false;

  Eigen::ConjugateGradient<MatrixReplacement<T>, Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
      cg_matrix_free_;
  Eigen::ConjugateGradient<MatrixX<T>, Eigen::Lower | Eigen::Upper> cg_dense_;

  int max_newton_iter_ = 500;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
