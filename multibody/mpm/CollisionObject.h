#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/mpm/AnalyticLevelSet.h"
#include "drake/multibody/mpm/SpatialVelocityTimeDependent.h"
#include "drake/multibody/math/spatial_algebra.h"

namespace drake {
namespace multibody {
namespace mpm {

// A class represents collision objects in MPM simulations. The collision
// objects are represented as analytical level sets (instead of particles, so
// they are not "MPM objects") with initial pose, spatial velocity (translation
// and angular velocities) and friction coefficient on the boundary. These
// collision objects are placed in a MPM simulation to simulate the environment
// that are not coupled with "MPM objects", for example, to enforce the boundary
// conditions prescribed by this object. We assume the collision object is not
// deformable.
template <typename T>
class CollisionObject {
 public:
    struct CollisionObjectState {
        math::RigidTransform<T> pose;
        std::unique_ptr<SpatialVelocityTimeDependent<T>> spatial_velocity;
    };

    CollisionObject(std::unique_ptr<AnalyticLevelSet> level_set,
                    CollisionObjectState initial_state,
                    T friction_coeff);

    // Advance the state (pose) of the collision object by dt
    void AdvanceOneTimeStep(T dt);

    // Apply the boundary condition prescribed by this object to a point in
    // the space with the given postion and velocity. On Boundaries,
    // we apply the Coulumb friction law. Given the velocity v, and denotes
    // its normal and tangential components by vₙ = (v ⋅ n)n, vₜ = v - vₙ,
    // the impulse of colliding with the wall is given by j = -m vₙ. The
    // Coulumb friction law states the amount of friction imposed is at most
    // μ ‖j‖, where \mu is the friction coefficient of the wall. If the normal
    // magnitude (v ⋅ n) of velocity is positive, i.e., the point is leaving
    // the object, we don't apply boundary condition. Otherwise,
    // If ‖vₜ‖ <= μ‖vₙ‖,   v_new = 0.0
    // Otherwise    ,   v_new = vₜ - μ‖vₙ‖t, t - tangential direction
    // Then we overwrite the passed in velocity with v_new.
    bool ApplyBoundaryCondition(const Vector3<T>& position,
                                Vector3<T>* velocity) const;

 private:
    // friend class CollisionObjectTest;

    // Given the normal vector and friction coefficient mu, update the input
    // velocity using Coulumb friction law
    void UpdateVelocityCoulumbFriction(const Vector3<T>& normal,
                                       Vector3<T>* velocity) const;

    CollisionObjectState state_;
    std::unique_ptr<AnalyticLevelSet> level_set_;
    T friction_coeff_;
};  // class CollisionObject

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
