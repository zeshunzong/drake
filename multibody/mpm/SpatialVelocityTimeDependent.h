#pragma once

#include <functional>
#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/math/spatial_algebra.h"

namespace drake {
namespace multibody {
namespace mpm {

// A base class providing the interface of time dependent spatial velocity
template <typename T>
class SpatialVelocityTimeDependent {
 public:
    SpatialVelocityTimeDependent();
    // Construct a time-independent spatial velocity
    SpatialVelocityTimeDependent(const multibody::SpatialVelocity<T>&
                                                            spatial_velocity);
    // Construct a time-dependent spatial velocity based on the given function
    SpatialVelocityTimeDependent(
            std::function<multibody::SpatialVelocity<T>(T)>
                                                    spatial_velocity_function);

    std::unique_ptr<SpatialVelocityTimeDependent> Clone() const {
        return std::make_unique<SpatialVelocityTimeDependent>(*this);
    }

    // Return the spatial velocity at time time_
    const multibody::SpatialVelocity<T>& GetSpatialVelocity() const;

    // Update the spatial velocity to be at time_ + dt, and update time_ = time_ + dt
    void AdvanceOneTimeStep(T dt);

    virtual ~SpatialVelocityTimeDependent() = default;

 protected:
    T time_ = 0.0;                                       // Current time of
                                                           // spatial velocity
    multibody::SpatialVelocity<T> spatial_velocity_;  // Spatial velocity
                                                           // at time T
    // The function v(t) that takes time as the argument and returns the
    // spatial velocity at time t.
    std::function<multibody::SpatialVelocity<T>(T)>
                                        spatial_velocity_function_;
};  // class SpatialVelocityTimeDependent

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
