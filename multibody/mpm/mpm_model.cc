#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {



template <typename T>
std::unique_ptr<MpmState<T>> MpmModel<T>::MakeMpmState(Particles& particles) const {
  return std::make_unique<MpmState<T>>(particles);
}

template <typename T>
std::unique_ptr<MpmState<T>> MpmModel<T>::MakeMpmState() const {
  return std::make_unique<MpmState<T>>();
}


}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::MpmModel<double>;
template class drake::multibody::mpm::MpmModel<drake::AutoDiffXd>;
