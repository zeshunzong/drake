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

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmModel {
 public:


  struct MaterialParameters {
        // Elastoplastic model of the object
        std::unique_ptr<ElastoPlasticModel> elastoplastic_model;
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

  
  virtual ~MpmModel() = default;



  /** Creates a default FemState compatible with this model. */
  std::unique_ptr<MpmState<T>> MakeMpmState(Particles& particles) const;

  std::unique_ptr<MpmState<T>> MakeMpmState() const;

  int num_particles_;


  /** Sets the gravity vector for all elements in this model. */
  void set_gravity_vector(const Vector3<T>& gravity) { gravity_ = gravity; }

  /** Returns the gravity vector for all elements in this model. */
  const Vector3<T>& gravity_vector() const { return gravity_; }


  void set_grid_h(const double h) {
    grid_h_ = h;
  }
  const double grid_h() const {
    return grid_h_;
  }

  void set_spatial_velocity(const multibody::SpatialVelocity<double>& v) {
    spatial_velocity_ = v;
  }
  const multibody::SpatialVelocity<double>& spatial_velocity() const {
    return spatial_velocity_;
  }
  void set_level_set(mpm::AnalyticLevelSet& s) {
    level_set_ = &s;
  }
  const mpm::AnalyticLevelSet* level_set() const {
    return level_set_;
  }
  void set_pose(math::RigidTransform<double>& pose){
    pose_ = &pose;
  }
  const math::RigidTransform<double>* pose() const {
    return pose_;
  }
  void set_material_params(MaterialParameters& material_params) {
    material_params_ = &material_params;
  }
  const MaterialParameters* material_params() const {
    return material_params_;
  }


  /** Constructs an empty MPM model. */
  MpmModel(){

  }


  systems::AbstractStateIndex particles_container_index_;
  double grid_h_;

 private:
  
  Vector3<T> gravity_{0, 0, -9.81};

  // parameters about geometry
  
  multibody::SpatialVelocity<double> spatial_velocity_{};
  mpm::AnalyticLevelSet* level_set_;
  math::RigidTransform<double>* pose_;
  MaterialParameters* material_params_;



};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
