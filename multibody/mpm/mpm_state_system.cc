#include "drake/multibody/mpm/mpm_state_system.h"
#include <iostream>
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
MpmStateSystem<T>::MpmStateSystem(const VectorX<T>& model_q,
                                  const VectorX<T>& model_v,
                                  const VectorX<T>& model_a,
                                  const Particles& particles) {
  num_dofs_ = model_q.size();
  DRAKE_THROW_UNLESS(model_q.size() == model_v.size());
  DRAKE_THROW_UNLESS(model_q.size() == model_a.size());
  DRAKE_THROW_UNLESS(model_q.size() % 3 == 0);
  q_index_ = this->DeclareDiscreteState(model_q);
  q0_index_ = this->DeclareDiscreteState(model_q);
  v_index_ = this->DeclareDiscreteState(model_v);
  a_index_ = this->DeclareDiscreteState(model_a);

  num_particles_ = particles.get_num_particles();
  particles_container_index_ = this->DeclareAbstractState(Value<Particles>(particles));
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::mpm::internal::MpmStateSystem);
