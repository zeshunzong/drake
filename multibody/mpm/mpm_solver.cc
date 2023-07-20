#include "drake/multibody/mpm/mpm_solver.h"

#include <algorithm>

#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {



template <typename T>
MpmSolver<T>::MpmSolver(const MpmModel<T>* model)
    : model_(model), grid_(model->grid_h_) {
  DRAKE_DEMAND(model_ != nullptr);
}

template <typename T>
MpmSolver<T>::MpmSolver(const MpmModel<T>* model, double dt)
    : model_(model), grid_(model->grid_h_),dt_(dt) {
  DRAKE_DEMAND(model_ != nullptr);
  std::cout << "construct solver with system dt="<< dt << std::endl; 

  // ------------------------ add BC -----------------
  // ground at z = 0
  multibody::SpatialVelocity<double> zero_velocity;
  zero_velocity.SetZero();
  std::unique_ptr<mpm::SpatialVelocityTimeDependent> left_hand_velocity_ptr =
      std::make_unique<mpm::SpatialVelocityTimeDependent>(zero_velocity);
  double left_hand_mu = 5.0;
  Vector3<double> left_hand_xscale = {10.0, 10.0, 10.0};
  std::unique_ptr<mpm::AnalyticLevelSet> left_hand_level_set =
                          std::make_unique<mpm::BoxLevelSet>(left_hand_xscale);
  Vector3<double> left_hand_translation = {0.0, 0.0, -10.0};
  math::RigidTransform<double> left_hand_pose =
                          math::RigidTransform<double>(left_hand_translation);
  collision_objects_.AddCollisionObject(std::move(left_hand_level_set), std::move(left_hand_pose),
                              std::move(left_hand_velocity_ptr), left_hand_mu);
  // ------------------------ add BC -----------------

}



template <typename T>
int MpmSolver<T>::AdvanceOneTimeStep(const MpmState<T>& prev_state,
                                     MpmState<T>* next_state) const {
  
  const Particles p_prev = prev_state.GetParticles(); //getchar();

  const std::vector<Vector3<double>>& positions_prev = p_prev.get_positions();

  Particles p_new(p_prev);
  p_new.ApplyPlasticityAndUpdateKirchhoffStresses();
  mpm_transfer_.SetUpTransfer(&grid_, &p_new);
  mpm_transfer_.TransferParticlesToGrid(p_new, &grid_);
  grid_.UpdateVelocity(dt_);

  // gravity, collision, boundary to be added
  Vector3<double> gravitational_acceleration{0.0,0.0,-9.8};
  grid_.ApplyGravitationalForces(dt_, gravitational_acceleration);
  // gravitational_force_.ApplyGravitationalForces(dt_, &grid_);

  collision_objects_.AdvanceOneTimeStep(dt_);
  grid_.EnforceBoundaryCondition(collision_objects_, dt_);

  mpm_transfer_.TransferGridToParticles(grid_, dt_, &p_new);
  p_new.AdvectParticles(dt_);

 
  // p_new.print_info();

  next_state->SetParticles(p_new);



  return 2;
}


template <typename T>
bool MpmSolver<T>::solver_converged(const T& residual_norm,
                                    const T& initial_residual_norm) const {
  return residual_norm < std::max(relative_tolerance_ * initial_residual_norm,
                                  absolute_tolerance_);
}



}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmSolver<double>;
