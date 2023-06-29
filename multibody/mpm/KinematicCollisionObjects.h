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
class KinematicCollisionObjects {
 public:
    KinematicCollisionObjects() = default;

    // Add a new collision object with the given intial conditions
    void AddCollisionObject(std::unique_ptr<AnalyticLevelSet> level_set,
                            math::RigidTransform<double> pose,
                            std::unique_ptr<SpatialVelocityTimeDependent>
                                                            spatial_velocity,
                            double friction_coeff);

    int get_num_collision_objects() const;

    // Advance the state of all collision objects by dt
    void AdvanceOneTimeStep(double dt);

    // Apply the boundary condition prescribed by all collision objects to a
    // point in the space with the given postion and velocity. For a grid point
    // locates in multiple collision objects, we impose boundary conditions
    // of every collision object, ordered as in collision_objects_, to this grid
    // point.
    // TODO(yiminlin.tri): May cause unexpected behavior at sharp corners
    bool ApplyBoundaryConditions(const Vector3<double>& position,
                                 Vector3<double>* velocity) const;

 private:
    std::vector<std::unique_ptr<CollisionObject>> collision_objects_{};
};  // class KinematicCollisionObjects

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
