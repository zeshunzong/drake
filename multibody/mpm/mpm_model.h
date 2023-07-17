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
#include "drake/multibody/mpm/ElastoPlasticModel.h"



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


  /** Constructs an empty MPM model. */
  MpmModel(){

  }


  systems::AbstractStateIndex particles_container_index_;

 private:
  /* The system that manages the states and cache entries of this MPM model.
   */

 
  
  Vector3<T> gravity_{0, 0, -9.81};
  
//   /* The Dirichlet boundary condition that the model is subject to. */
//   internal::DirichletBoundaryCondition<T> dirichlet_bc_;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
