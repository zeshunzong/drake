#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include <Eigen/Sparse>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
// #include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/mpm/mpm_state.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/mpm/AnalyticLevelSet.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/math/spatial_velocity.h"
#include "drake/multibody/mpm/KinematicCollisionObjects.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmModel {
 public:
  struct MaterialParameters {
        // Elastoplastic model of the object
        std::unique_ptr<ElastoPlasticModel<double>> elastoplastic_model;
        // @pre density is positive
        // Density and the initial velocity of the object, we assume the object
        // has uniform density and velocity.
        double density;
        // V_WB, The object B's spatial velocity measured and expressed in the
        // world frame W.
        multibody::SpatialVelocity<double> initial_velocity;
        // User defined parameter to control the minimum number of particles per
        // grid cell.
        int min_num_particles_per_cell;
    };

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmModel);

    /** Constructs an empty MPM model. */
  MpmModel(){}

  virtual ~MpmModel() = default;



  /** Creates a default FemState compatible with this model. */
  std::unique_ptr<MpmState<T>> MakeMpmState(Particles& particles) const{
    return std::make_unique<MpmState<T>>(particles);
  }

  std::unique_ptr<MpmState<T>> MakeMpmState() const {
    return std::make_unique<MpmState<T>>();
  }

  void set_grid_h(const double h) {
    grid_h_ = h;
  }
  double grid_h() const {
    return grid_h_;
  }

  systems::AbstractStateIndex particles_container_index_;

  int num_particles_;
  double grid_h_;
  multibody::SpatialVelocity<double> object_initial_velocity_{};
  MaterialParameters mp_{};
  std::unique_ptr<mpm::AnalyticLevelSet> level_set_;
  math::RigidTransform<double> pose_;

 private:
  
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
