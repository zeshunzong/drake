#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MpmModel<T>::Builder::Build(const Particles& particles) {
  ThrowIfBuilt();
  model_->UpdateMpmStateSystem(particles);
  built_ = true;
}

template <typename T>
void MpmModel<T>::Builder::ThrowIfBuilt() const {
  if (built_) {
    throw std::logic_error(
        "Build() has been called on this Builder. Create a new Builder if you "
        "need to add more elements to the FEM model.");
  }
}

template <typename T>
std::unique_ptr<MpmState<T>> MpmModel<T>::MakeMpmState() const {
  return std::make_unique<MpmState<T>>(mpm_state_system_.get());
}



template <typename T>
MpmModel<T>::MpmModel()
    : mpm_state_system_(std::make_unique<internal::MpmStateSystem<T>>(
          VectorX<T>(0), VectorX<T>(0), VectorX<T>(0), Particles())) {}

template <typename T>
void MpmModel<T>::ThrowIfModelStateIncompatible(
    const char* func, const MpmState<T>& mpm_state) const {
  if (!mpm_state.is_created_from_system(*mpm_state_system_)) {
    throw std::logic_error(std::string(func) +
                           "(): The FEM model and state are not compatible.");
  }
}

template <typename T>
void MpmModel<T>::UpdateMpmStateSystem(const Particles& particles) {
  VectorX<T> model_positions = GetReferencePositions();
  VectorX<T> model_velocities = VectorX<T>::Zero(model_positions.size());
  VectorX<T> model_accelerations = VectorX<T>::Zero(model_positions.size());
  mpm_state_system_ = std::make_unique<internal::MpmStateSystem<T>>(
      model_positions, model_velocities, model_accelerations, particles);
  // DeclareCacheEntries(mpm_state_system_.get());
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::MpmModel<double>;
template class drake::multibody::mpm::MpmModel<drake::AutoDiffXd>;
