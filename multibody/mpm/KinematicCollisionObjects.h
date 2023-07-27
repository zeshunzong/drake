#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/mpm/CollisionObject.h"

namespace drake {
namespace multibody {
namespace mpm {

// A class holding all collision objects' information in a MPM simulation.
template <typename T>
class KinematicCollisionObjects {
 public:
    KinematicCollisionObjects() = default;

    // Add a new collision object with the given intial conditions
    void AddCollisionObject(std::unique_ptr<AnalyticLevelSet> level_set,
                            math::RigidTransform<T> pose,
                            std::unique_ptr<SpatialVelocityTimeDependent<T>>
                                                            spatial_velocity,
                            T friction_coeff);

    int get_num_collision_objects() const;

    // Advance the state of all collision objects by dt
    void AdvanceOneTimeStep(T dt);

    // Apply the boundary condition prescribed by all collision objects to a
    // point in the space with the given postion and velocity. For a grid point
    // locates in multiple collision objects, we impose boundary conditions
    // of every collision object, ordered as in collision_objects_, to this grid
    // point.
    // TODO(yiminlin.tri): May cause unexpected behavior at sharp corners
    bool ApplyBoundaryConditions(const Vector3<T>& position,
                                 Vector3<T>* velocity) const;

 private:
    std::vector<std::unique_ptr<CollisionObject<T>>> collision_objects_{};
};  // class KinematicCollisionObjects

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
