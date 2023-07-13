#include "drake/multibody/mpm/mpm_solver.h"

#include <algorithm>

#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {



template <typename T>
MpmSolver<T>::MpmSolver(const MpmModel<T>* model)
    : model_(model) {
  DRAKE_DEMAND(model_ != nullptr);
}

template <typename T>
int MpmSolver<T>::AdvanceOneTimeStep(const MpmState<T>& prev_state,
                                     MpmState<T>* next_state) const {
  

  const Particles p_prev = prev_state.GetParticles();

  const std::vector<Vector3<double>>& positions_prev = p_prev.get_positions();

  

  Particles p_new(1);
  p_new.set_positions(positions_prev);

  
  p_new.advect_x_coord();

  // const std::vector<Vector3<double>>& positions_new = p_new.get_positions();
  // std::cout << "print old positions in mpm solver" << std::endl;
  // std::cout << positions_prev[0][0]<< " , " << positions_prev[0][1]<< "," <<positions_prev[0][2] << std::endl;
  // std::cout << "print new positions in mpm solver" << std::endl;
  // std::cout << positions_new[0][0]<< " , " << positions_new[0][1]<< "," <<positions_new[0][2] << std::endl;
  // getchar();
  // to be fixed

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
