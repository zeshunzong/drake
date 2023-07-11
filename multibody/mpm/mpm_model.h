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
// #include "drake/multibody/fem/dirichlet_boundary_condition.h"
#include "drake/multibody/mpm/mpm_state.h"
// #include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmModel {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmModel);

  /** %Builder that builds the MpmModel. Each concrete %MpmModel must define its
   own builder, subclassing this class, to add new elements to the model. */
  class Builder {
   public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Builder);

    virtual ~Builder() = default;


    void Build(VectorX<T> reference_positions);

   
    /** Throws an exception if Build() has been called on this %Builder. */
    void ThrowIfBuilt() const;

    /** Constructs a new builder that builds into the given `model`.
     @pre model != nullptr.
     @note The `model` pointer is persisted and the pointed to MpmModel must
     outlive `this` *Builder. */
    explicit Builder(MpmModel<T>* model) : model_{model} {
      DRAKE_DEMAND(model_ != nullptr);
    }

    void DoBuild(const VectorX<T> reference_positions){
      std::cout << "temporary setting reference solution of MPMModel in builder" << std::endl;
      model_->reference_positions_.conservativeResize(reference_positions.size());
      for (int i=0; i < reference_positions.size(); ++i) {
        model_->reference_positions_[0] = reference_positions[i];
      }
    }

   private:
    /* The model that `this` builder builds into. */
    MpmModel<T>* model_{nullptr};
    /* Flag to keep track of whether Build() has been called on this builder. */
    bool built_{false};
  };

  virtual ~MpmModel() = default;

  /* The `num_dofs()` is always a multiple of 3. It is enforced by
    MpmStateSystem. */
  /** The number of nodes that are associated with this model. */
  int num_nodes() const { return num_dofs() / 3; }

  /** The number of degrees of freedom in this model. */
  int num_dofs() const { return mpm_state_system_->num_dofs(); }


  /** Creates a default FemState compatible with this model. */
  std::unique_ptr<MpmState<T>> MakeMpmState() const;

//   /** Calculates the residual G(x, v, a) (see class doc) evaluated at the
//    given FEM state. The residual for degrees of freedom with Dirichlet boundary
//    conditions is set to zero. Therefore their residual should not be used as a
//    metric for the error on the boundary condition.
//    @pre residual != nullptr.
//    @throws std::exception if the FEM state is incompatible with this model. */
//   void CalcResidual(const FemState<T>& fem_state,
//                     EigenPtr<VectorX<T>> residual) const;



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
  MpmModel();


  /** Updates the system that manages the states and the cache entries of this
   FEM model. Must be called before calling MakeFemState() after the FEM model
   changes (e.g. adding new elements). */
  void UpdateMpmStateSystem();

//   /** Derived classes should override this method to declare cache entries in
//    the given `fem_state_system`. */
//   virtual void DeclareCacheEntries(
//       internal::FemStateSystem<T>* mpm_state_system) = 0;

  /** Returns the FemStateSystem that manages the states and cache entries in
   this %MpmModel. */
  const internal::MpmStateSystem<T>& mpm_state_system() const {
    return *mpm_state_system_;
  }

 private:
  /* The system that manages the states and cache entries of this MPM model.
   */

  VectorX<T> GetReferencePositions() const {
    return reference_positions_;
  }
  std::unique_ptr<internal::MpmStateSystem<T>> mpm_state_system_;
  Vector3<T> gravity_{0, 0, -9.81};
  VectorX<T> reference_positions_{};
//   /* The Dirichlet boundary condition that the model is subject to. */
//   internal::DirichletBoundaryCondition<T> dirichlet_bc_;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
