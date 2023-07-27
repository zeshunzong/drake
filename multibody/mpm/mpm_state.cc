#include "drake/multibody/mpm/mpm_state.h"

namespace drake {
namespace multibody {
namespace mpm {

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

template class MpmState<double>;
template class MpmState<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
