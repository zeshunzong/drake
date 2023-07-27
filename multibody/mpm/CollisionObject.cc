#include "drake/multibody/mpm/CollisionObject.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
CollisionObject<T>::CollisionObject(std::unique_ptr<AnalyticLevelSet<T>> level_set,
                    CollisionObject<T>::CollisionObjectState initial_state,
                    T friction_coeff): state_(std::move(initial_state)),
                                            level_set_(std::move(level_set)),
                                            friction_coeff_(friction_coeff) {}

template <typename T>
void CollisionObject<T>::AdvanceOneTimeStep(T dt) {
    // Update spatial velocity
    state_.spatial_velocity->AdvanceOneTimeStep(dt);
    // Updated translation velocity
    Vector3<T> translation_new = state_.pose.translation()
            +dt*state_.spatial_velocity->GetSpatialVelocity().translational();
    // Angular velocity
    const Vector3<T>& omega =
                    state_.spatial_velocity->GetSpatialVelocity().rotational();
    Matrix3<T> angular_velocity_matrix, R_new, S;
    angular_velocity_matrix <<       0.0, -omega(2),  omega(1),
                                omega(2),       0.0, -omega(0),
                               -omega(1),  omega(0),       0.0;
    Matrix3<T> R = (Matrix3<T>::Identity()
                        +dt*angular_velocity_matrix)
                        *state_.pose.rotation().matrix();
    // Orthogonalize the new rotation matrix via polar decomposition
    fem::internal::PolarDecompose<T>(R, &R_new, &S);
    drake::math::RotationMatrix<T> rotation_new =
                                    drake::math::RotationMatrix<T>(R_new);
    state_.pose.set(rotation_new, translation_new);
}

// TODO(yiminlin.tri): return true if applied BC
template <typename T>
bool CollisionObject<T>::ApplyBoundaryCondition(
                                        const Vector3<T>& position,
                                        Vector3<T>* velocity) const {
    // Get the translational component of the relative spatial velocity at point
    // p_WQ (the grid point) between the grid point Q and the collision object R
    // by subtracting the translational component of the spatial velocity of a
    // point (Rq) coincident with p_WQ on the collision object R from the
    // translational velocity of the grid point, v_WQ (`velocity` on input).
    // Ro, Rq denotes the frame centered at the collision object and tthe grid
    // point respectively.
    const Vector3<T> p_WQ = position;
    Vector3<T> v_WQ = *velocity;

    // The pose and spatial velocity of the collision object in world frame.
    const math::RigidTransform<T>& X_WR = state_.pose;
    const multibody::SpatialVelocity<T>& V_WR = state_.spatial_velocity
                                                        ->GetSpatialVelocity();

    // Rotation matrix from world to the collision object
    const math::RotationMatrix<T> Rot_RW = X_WR.rotation().inverse();

    // Express the relative position in the world frame W
    const Vector3<T> p_RoRq_W = p_WQ - X_WR.translation();

    // Express the relative position in the collision object's frame R.
    const Vector3<T> p_RoRq_R = Rot_RW * p_RoRq_W;

    // Don't apply BC if the input grid point is not in the collision object
    if (!level_set_->InInterior(p_RoRq_R)) return false;

    // Compute the spatial velocity at Rq for the collision object.
    const multibody::SpatialVelocity<T> V_WRq = V_WR.Shift(p_RoRq_W);

    // Compute the relative (translational) velocity of grid node
    // Q relative to Frame Rq, expressed in the world frame.
    Vector3<T> v_RqQ_W = v_WQ - V_WRq.translational();

    // Express the relative velocity in the collision object's frame R.
    Vector3<T> v_RqQ_R = Rot_RW * v_RqQ_W;

    UpdateVelocityCoulumbFriction(level_set_->Normal(p_RoRq_R), &v_RqQ_R);

    // Transform the velocity back to the world frame.
    v_RqQ_W = X_WR.rotation() * v_RqQ_R;
    v_WQ = v_RqQ_W + V_WRq.translational();
    *velocity = v_WQ;
    return true;
}

template <typename T>
void CollisionObject<T>::UpdateVelocityCoulumbFriction(
                                            const Vector3<T>& n,
                                            Vector3<T>* velocity) const {
    // If the velocity is moving out from the object, we don't apply the
    // friction
    T vdotn = velocity->dot(n);
    if (vdotn > 0)  return;
    // Normal and tangential components of the velocity
    Vector3<T> vn = vdotn*n;
    Vector3<T> vt = *velocity - vn;
    // Normal and tangential speed
    T vn_norm = vn.norm();
    T vt_norm = vt.norm();

    // Limit the amount of friction to at most eliminating the
    // tangential velocity
    if (vt_norm <= friction_coeff_*vn_norm) {
        *velocity = Vector3<T>::Zero();
    } else {
        // If the tangential velocity is zero, the updated velocity is
        // zero by above.
        *velocity = vt - friction_coeff_*vn_norm*vt/vt_norm;
    }
}


template class CollisionObject<double>;
template class CollisionObject<AutoDiffXd>;


}  // namespace mpm
}  // namespace multibody
}  // namespace drake
