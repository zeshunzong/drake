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

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmModel {
 public:
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

//   /** Applies boundary condition set for this %MpmModel to the input `state`.
//    No-op if no boundary condition is set.
//    @pre fem_state != nullptr.
//    @throws std::exception if the FEM state is incompatible with this model. */
//   void ApplyBoundaryCondition(FemState<T>* fem_state) const;

//   // TODO(xuchenhan-tri): Internal object in public signature in non-internal
//   //  class.
//   /** Sets the Dirichlet boundary condition that this model is subject to. */
//   void SetDirichletBoundaryCondition(
//       internal::DirichletBoundaryCondition<T> dirichlet_bc) {
//     dirichlet_bc_ = std::move(dirichlet_bc);
//   }

//   /** Returns the Dirichlet boundary condition that this model is subject to. */
//   const internal::DirichletBoundaryCondition<T>& dirichlet_boundary_condition()
//       const {
//     return dirichlet_bc_;
//   }

  /** (Internal use only) Throws std::exception to report a mismatch between
  the FEM model and state that were passed to API method `func`. */
  void ThrowIfModelStateIncompatible(const char* func,
                                     const MpmState<T>& mpm_state) const;


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
