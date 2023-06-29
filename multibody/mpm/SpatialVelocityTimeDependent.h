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
class SpatialVelocityTimeDependent {
 public:
    SpatialVelocityTimeDependent();
    // Construct a time-independent spatial velocity
    SpatialVelocityTimeDependent(const multibody::SpatialVelocity<double>&
                                                            spatial_velocity);
    // Construct a time-dependent spatial velocity based on the given function
    SpatialVelocityTimeDependent(
            std::function<multibody::SpatialVelocity<double>(double)>
                                                    spatial_velocity_function);

    std::unique_ptr<SpatialVelocityTimeDependent> Clone() const {
        return std::make_unique<SpatialVelocityTimeDependent>(*this);
    }

    // Return the spatial velocity at time T_
    const multibody::SpatialVelocity<double>& GetSpatialVelocity() const;

    // Update the spatial velocity to be at T_ + dt, and update T_ = T_ + dt
    void AdvanceOneTimeStep(double dt);

    virtual ~SpatialVelocityTimeDependent() = default;

 protected:
    double T_ = 0.0;                                       // Current time of
                                                           // spatial velocity
    multibody::SpatialVelocity<double> spatial_velocity_;  // Spatial velocity
                                                           // at time T
    // The function v(t) that takes time as the argument and returns the
    // spatial velocity at time t.
    std::function<multibody::SpatialVelocity<double>(double)>
                                        spatial_velocity_function_;
};  // class SpatialVelocityTimeDependent

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
