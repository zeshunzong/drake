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



// template <typename T>
// MpmModel<T>::MpmModel()
//     : mpm_state_system_(std::make_unique<internal::MpmStateSystem<T>>(
//           VectorX<T>(0), VectorX<T>(0), VectorX<T>(0), Particles())) {}

template <typename T>
void MpmModel<T>::ThrowIfModelStateIncompatible(
    const char* func, const MpmState<T>& mpm_state) const {
  // if (!mpm_state.is_created_from_system(*mpm_state_system_)) {
  //   throw std::logic_error(std::string(func) +
  //                          "(): The FEM model and state are not compatible.");
  // }
}



}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::MpmModel<double>;
template class drake::multibody::mpm::MpmModel<drake::AutoDiffXd>;
