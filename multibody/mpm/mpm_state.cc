#include "drake/multibody/mpm/mpm_state.h"

namespace drake {
namespace multibody {
namespace mpm {

// template <typename T>
// MpmState<T>::MpmState(const internal::MpmStateSystem<T>* system)
//     : system_(system) {
//   DRAKE_DEMAND(system != nullptr);
//   owned_context_ = system_->CreateDefaultContext();
// }

// template <typename T>
// MpmState<T>::MpmState(const internal::MpmStateSystem<T>* system,
//                       const systems::Context<T>* context)
//     : system_(system), context_(context) {
//   DRAKE_DEMAND(system != nullptr);
//   DRAKE_DEMAND(context != nullptr);
//   system->ValidateContext(*context);
// }

template <typename T>
std::unique_ptr<MpmState<T>> MpmState<T>::Clone() const {
  // if (owned_context_ != nullptr) {
  //   auto clone = std::make_unique<MpmState<T>>(this->system_);
  //   clone->owned_context_->SetTimeStateAndParametersFrom(*this->owned_context_);
  //   return clone;
  // }
  // DRAKE_DEMAND(context_ != nullptr);
  // Note that this creates a "shared context" clone. See the class
  // documentation for cautionary notes.
  return std::make_unique<MpmState<T>>(this->particles_);
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::mpm::MpmState);
