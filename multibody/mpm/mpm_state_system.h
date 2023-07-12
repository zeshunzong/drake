#pragma once

#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/mpm/Particles.h"


namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* A leaf system that manages state and element data in a MPM model.
 @tparam_nonsymbolic_scalar */
template <typename T>
class MpmStateSystem : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmStateSystem);

  /* Constructs a new MpmStateSystem with the given model states. No element
   data is declared.
   @pre model_q, model_v, model_a all have the same size.
   @pre model_q's size is a multiple of 3. */
  MpmStateSystem(const VectorX<T>& model_q, const VectorX<T>& model_v,
                 const VectorX<T>& model_a,
                 const Particles& particles);

  /* Promotes DeclareCacheEntry so that MpmStateSystem can declare cache
  entries publicly. */
  using systems::SystemBase::DeclareCacheEntry;

  /* Returns the discrete state index. */
  systems::DiscreteStateIndex mpm_position_index() const { return q_index_; }
  systems::DiscreteStateIndex mpm_previous_step_position_index() const {
    return q0_index_;
  }
  systems::DiscreteStateIndex mpm_velocity_index() const { return v_index_; }
  systems::DiscreteStateIndex mpm_acceleration_index() const {
    return a_index_;
  }
  systems::AbstractStateIndex particles_container_index() const {
    return particles_container_index_;
  }

  /* Returns the number of degrees of freedom in the system. */
  int num_dofs() const { return num_dofs_; }

  

 private:
  systems::DiscreteStateIndex q_index_;
  systems::DiscreteStateIndex q0_index_;
  systems::DiscreteStateIndex v_index_;
  systems::DiscreteStateIndex a_index_;
  int num_dofs_{0};

  int num_particles_{0};
  systems::AbstractStateIndex particles_container_index_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
