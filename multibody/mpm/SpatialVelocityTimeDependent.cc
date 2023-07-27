#include "drake/multibody/mpm/SpatialVelocityTimeDependent.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
SpatialVelocityTimeDependent<T>::SpatialVelocityTimeDependent(): time_(0) { }

template <typename T>
SpatialVelocityTimeDependent<T>::SpatialVelocityTimeDependent(
                    const multibody::SpatialVelocity<T>& spatial_velocity):
                                time_(0.0), spatial_velocity_(spatial_velocity) {
    spatial_velocity_function_ =
                [this](T)->multibody::SpatialVelocity<T> {
                                        return spatial_velocity_;  };
}

template <typename T>
SpatialVelocityTimeDependent<T>::SpatialVelocityTimeDependent(
            std::function<multibody::SpatialVelocity<T>(T)>
                                        spatial_velocity_function): time_(0.0),
                        spatial_velocity_function_(spatial_velocity_function) {
    spatial_velocity_ = spatial_velocity_function_(0.0);
}

template <typename T>
const multibody::SpatialVelocity<T>&
                    SpatialVelocityTimeDependent<T>::GetSpatialVelocity() const {
    return spatial_velocity_;
}

template <typename T>
void SpatialVelocityTimeDependent<T>::AdvanceOneTimeStep(T dt) {
    time_ += dt;
    spatial_velocity_ = spatial_velocity_function_(time_);
}

template class SpatialVelocityTimeDependent<double>;
template class SpatialVelocityTimeDependent<AutoDiffXd>;


}  // namespace mpm
}  // namespace multibody
}  // namespace drake

