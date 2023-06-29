#include "drake/multibody/mpm/SpatialVelocityTimeDependent.h"

namespace drake {
namespace multibody {
namespace mpm {

SpatialVelocityTimeDependent::SpatialVelocityTimeDependent(): T_(0.0) { }

SpatialVelocityTimeDependent::SpatialVelocityTimeDependent(
                    const multibody::SpatialVelocity<double>& spatial_velocity):
                                T_(0.0), spatial_velocity_(spatial_velocity) {
    spatial_velocity_function_ =
                [this](double)->multibody::SpatialVelocity<double> {
                                        return spatial_velocity_;  };
}

SpatialVelocityTimeDependent::SpatialVelocityTimeDependent(
            std::function<multibody::SpatialVelocity<double>(double)>
                                        spatial_velocity_function): T_(0.0),
                        spatial_velocity_function_(spatial_velocity_function) {
    spatial_velocity_ = spatial_velocity_function_(0.0);
}

const multibody::SpatialVelocity<double>&
                    SpatialVelocityTimeDependent::GetSpatialVelocity() const {
    return spatial_velocity_;
}

void SpatialVelocityTimeDependent::AdvanceOneTimeStep(double dt) {
    T_ += dt;
    spatial_velocity_ = spatial_velocity_function_(T_);
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

