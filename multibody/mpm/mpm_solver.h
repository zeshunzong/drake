#pragma once

#include <memory>
#include <iostream>
#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/mpm/mpm_model.h"
#include "drake/multibody/mpm/SparseGrid.h"
#include "drake/multibody/mpm/MPMTransfer.h"
#include "drake/multibody/mpm/KinematicCollisionObjects.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
class MpmSolverScratchData {
 public:
  //DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmSolverScratchData);

  MpmSolverScratchData(double grid_h): grid_(grid_h) {}

  SparseGrid<double> grid_;
  MPMTransfer mpm_transfer_;

};


template <typename T>
class MpmSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmSolver);

//  struct MpmScratch {
//    SparseGrid grid_;
//    MPMTransfer mpm_transfer_;
//  };

 
  MpmSolver(const MpmModel<T>* model);

  MpmSolver(const MpmModel<T>* model, double dt);

  MpmSolver(const MpmModel<T>* model, double dt, KinematicCollisionObjects<double> collision_objects);

  int AdvanceOneTimeStep(const MpmState<T>& prev_state,  MpmState<T>* next_state, MpmSolverScratchData<T>* scratch) const;

  /* Returns the FEM model that this solver solves for. */
  const MpmModel<T>& model() const { return *model_; }


  /* Sets the relative tolerance, unitless. See solver_converged() for how
   the tolerance is used. The default value is 1e-4. */
  void set_relative_tolerance(double tolerance) {
    relative_tolerance_ = tolerance;
  }

  double relative_tolerance() const { return relative_tolerance_; }

  /* Sets the absolute tolerance with unit Newton. See solver_converged() for
   how the tolerance is used. The default value is 1e-6. */
  void set_absolute_tolerance(double tolerance) {
    absolute_tolerance_ = tolerance;
  }

  double absolute_tolerance() const { return absolute_tolerance_; }

  /* The solver is considered as converged if ‖r‖ <= max(εᵣ * ‖r₀‖, εₐ) where r
   and r₀ are `residual_norm` and `initial_residual_norm` respectively, and εᵣ
   and εₐ are relative and absolute tolerance respectively. */
  bool solver_converged(const T& residual_norm,
                        const T& initial_residual_norm) const;

  double get_dt() const {return dt_;}
  void set_dt(const double dt) {dt_ = dt;}

  double get_grid_h() const {return grid_h_;}
  void set_grid_h(const double grid_h) {grid_h_ = grid_h;}


  
  

  mutable KinematicCollisionObjects<double> collision_objects_{};

 private:

  /* The FEM model being solved by `this` solver. */
  const MpmModel<T>* model_{nullptr};

  // Run the simulation with timestep size dt till endtime
  double dt_; 
  // Grid parameters, as documented in SparseGrid Class
  double grid_h_;


  /* The discrete time integrator the solver uses. */
  /* Tolerance for convergence. */
  double relative_tolerance_{1e-4};  // unitless.
  double absolute_tolerance_{1e-6};  // unit N.
  /* Max number of Newton-Raphson iterations the solver takes before it gives
   up. */
  int kMaxIterations_{100};
  // mutable SparseGrid grid_;
  // mutable MPMTransfer mpm_transfer_;
  
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
