#include "drake/multibody/mpm/KinematicCollisionObjects.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void KinematicCollisionObjects<T>::AddCollisionObject(
                    std::unique_ptr<AnalyticLevelSet<T>> level_set,
                    math::RigidTransform<T> pose,
                    std::unique_ptr<SpatialVelocityTimeDependent<T>>
                                                      spatial_velocity,
                    T friction_coeff) {
    typename CollisionObject<T>::CollisionObjectState state = {std::move(pose),
                                                   std::move(spatial_velocity)};
    collision_objects_.emplace_back(
                        std::make_unique<CollisionObject<T>>(std::move(level_set),
                                                          std::move(state),
                                                          friction_coeff));
}

template <typename T>
int KinematicCollisionObjects<T>::get_num_collision_objects() const {
    return collision_objects_.size();
}

template <typename T>
void KinematicCollisionObjects<T>::AdvanceOneTimeStep(T dt) {
    for (auto& obj : collision_objects_) {
        obj->AdvanceOneTimeStep(dt);
    }
}

template <typename T>
bool KinematicCollisionObjects<T>::ApplyBoundaryConditions(
                                            const Vector3<T>& position,
                                            Vector3<T>* velocity) const {
    bool applied_BC = false;
    for (const auto& obj : collision_objects_) {
        bool applied = obj->ApplyBoundaryCondition(position, velocity);
        applied_BC = applied_BC || applied;
    }
    return applied_BC;
}

template class KinematicCollisionObjects<double>;
template class KinematicCollisionObjects<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
