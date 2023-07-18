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
}

template <typename T>
int MpmSolver<T>::AdvanceOneTimeStep(const MpmState<T>& prev_state,
                                     MpmState<T>* next_state) const {
  
  std::cout << "dt is " << dt_ << std::endl;                                    
  const Particles p_prev = prev_state.GetParticles(); //getchar();

  const std::vector<Vector3<double>>& positions_prev = p_prev.get_positions();

  Particles p_new(p_prev);
  p_new.ApplyPlasticityAndUpdateKirchhoffStresses();
  mpm_transfer_.SetUpTransfer(&grid_, &p_new);
  mpm_transfer_.TransferParticlesToGrid(p_new, &grid_);
  grid_.UpdateVelocity(dt_);

  // gravity, collision, boundary to be added
  std::cout << "gravity, collision, boundary to be added" << std::endl;

  mpm_transfer_.TransferGridToParticles(grid_, dt_, &p_new);
  p_new.AdvectParticles(dt_);

  // Particles p_new(p_prev.get_num_particles());
  // p_new.set_positions(positions_prev);
  // p_new.advect_x_coord();
  p_new.print_info();

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
