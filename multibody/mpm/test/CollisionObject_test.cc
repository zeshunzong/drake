#include "drake/multibody/fem/mpm-dev/CollisionObject.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/fem/mpm-dev/AnalyticLevelSet.h"
#include "drake/multibody/fem/mpm-dev/SpatialVelocityTimeDependent.h"

namespace drake {
namespace multibody {
namespace mpm {

constexpr double TOLERANCE = 1e-10;

class CollisionObjectTest : public ::testing::Test {
 protected:
    void SetUp() {
        // Construct a half space with normal (1, 1, 1), and pass through point
        // (2, 2, 2)
        double half_space_mu = 0.1;
        Vector3<double> half_space_normal = Vector3<double>(1, 1, 1);
        Vector3<double> half_space_translation = Vector3<double>(2, 2, 2);
        math::RigidTransform<double> half_space_pose
                        = math::RigidTransform<double>(half_space_translation);
        multibody::SpatialVelocity<double> half_space_v;
        half_space_v.SetZero();
        std::unique_ptr<SpatialVelocityTimeDependent> half_space_v_ptr
                = std::make_unique<SpatialVelocityTimeDependent>(half_space_v);
        std::unique_ptr<HalfSpaceLevelSet> half_space_level_set =
                        std::make_unique<HalfSpaceLevelSet>(half_space_normal);
        CollisionObject::CollisionObjectState half_space_state =
                                                {half_space_pose,
                                                 std::move(half_space_v_ptr)};
        half_space_ = std::make_unique<CollisionObject>(
                                            std::move(half_space_level_set),
                                            std::move(half_space_state),
                                            half_space_mu);

        // Construct a moving box with size (1, 1, 1), and centered at (2, 2, 2)
        // with linear velocity (1, 0, 0)
        double box_mu = 0.2;
        Vector3<double> box_xscale = Vector3<double>(1, 1, 1);
        Vector3<double> box_translation = Vector3<double>(2, 2, 2);
        math::RigidTransform<double> box_pose
                        = math::RigidTransform<double>(box_translation);
        multibody::SpatialVelocity<double> box_v;
        box_v.translational() = Vector3<double>{1.0, 0.0, 0.0};
        box_v.rotational() = Vector3<double>::Zero();
        std::unique_ptr<SpatialVelocityTimeDependent> box_v_ptr
                    = std::make_unique<SpatialVelocityTimeDependent>(box_v);
        std::unique_ptr<BoxLevelSet> box_level_set =
                                std::make_unique<BoxLevelSet>(box_xscale);
        CollisionObject::CollisionObjectState box_state =
                                                       {box_pose,
                                                        std::move(box_v_ptr)};
        box_ = std::make_unique<CollisionObject>(std::move(box_level_set),
                                                 std::move(box_state), box_mu);

        // Construct a rotating sphere with radius 1, centered at (0, 0, 0)
        // with angular velocity (pi, 0, 0)
        double sphere_mu = 0.2;
        double sphere_radius = 1.0;
        Vector3<double> sphere_translation = Vector3<double>(0, 0, 0);
        math::RigidTransform<double> sphere_pose
                        = math::RigidTransform<double>(sphere_translation);
        multibody::SpatialVelocity<double> sphere_v;
        sphere_v.translational() = Vector3<double>::Zero();
        sphere_v.rotational() = Vector3<double>{M_PI, 0.0, 0.0};
        std::unique_ptr<SpatialVelocityTimeDependent> sphere_v_ptr
                    = std::make_unique<SpatialVelocityTimeDependent>(sphere_v);
        std::unique_ptr<SphereLevelSet> sphere_level_set =
                            std::make_unique<SphereLevelSet>(sphere_radius);
        CollisionObject::CollisionObjectState sphere_state =
                                                    {sphere_pose,
                                                     std::move(sphere_v_ptr)};
        sphere_ = std::make_unique<CollisionObject>(std::move(sphere_level_set),
                                                    std::move(sphere_state),
                                                    sphere_mu);

        // Construct a moving and rotating sphere with radius 1, centered at (0,
        // 0, 0) with angular velocity (pi, 0, 0) and translational velocity (1,
        // 0, 0)
        double moving_sphere_mu = 0.2;
        double moving_sphere_radius = 1.0;
        Vector3<double> moving_sphere_translation = Vector3<double>(0, 0, 0);
        math::RigidTransform<double> moving_sphere_pose
                    = math::RigidTransform<double>(moving_sphere_translation);
        multibody::SpatialVelocity<double> moving_sphere_v;
        moving_sphere_v.translational() = Vector3<double>{1.0, 0.0, 0.0};
        moving_sphere_v.rotational() = Vector3<double>{M_PI, 0.0, 0.0};
        std::unique_ptr<SpatialVelocityTimeDependent> moving_sphere_v_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(moving_sphere_v);
        std::unique_ptr<SphereLevelSet> moving_sphere_level_set =
                        std::make_unique<SphereLevelSet>(moving_sphere_radius);
        CollisionObject::CollisionObjectState moving_sphere_state =
                                        {moving_sphere_pose,
                                         std::move(moving_sphere_v_ptr)};
        moving_sphere_ = std::make_unique<CollisionObject>(
                                        std::move(moving_sphere_level_set),
                                        std::move(moving_sphere_state),
                                        moving_sphere_mu);

        // Construct a moving cylindrical boundary with central axis parallel to
        // y-axis, centered at (0.0, 2.0, 3.0), and moving with velocity (1, 0,
        // 0). The height and radius of the cylinder are 2 and 1.
        double moving_cylinder_mu = 0.1;
        double moving_cylinder_h = 2.0;
        double moving_cylinder_r = 1.0;
        math::RollPitchYaw moving_cylinder_rpw = {M_PI/2.0, 0.0, 0.0};
        Vector3<double> moving_cylinder_translation = Vector3<double>(0, 2, 3);
        math::RigidTransform<double> moving_cylinder_pose =
                                                {moving_cylinder_rpw,
                                                 moving_cylinder_translation};
        multibody::SpatialVelocity<double> moving_cylinder_v;
        moving_cylinder_v.translational() = Vector3<double>{1.0, 0.0, 0.0};
        moving_cylinder_v.rotational() = Vector3<double>::Zero();
        std::unique_ptr<SpatialVelocityTimeDependent> moving_cylinder_v_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(moving_cylinder_v);
        std::unique_ptr<CylinderLevelSet> moving_cylinder_level_set
                    = std::make_unique<CylinderLevelSet>(moving_cylinder_h,
                                                         moving_cylinder_r);
        CollisionObject::CollisionObjectState cylinder_state =
                                            {moving_cylinder_pose,
                                             std::move(moving_cylinder_v_ptr)};
        moving_cylinder_ = std::make_unique<CollisionObject>(
                                        std::move(moving_cylinder_level_set),
                                        std::move(cylinder_state),
                                        moving_cylinder_mu);
    }

    void TestBoxAdvanceOneTimeStep() {
        // Check the current pose of the box
        ASSERT_TRUE(CompareMatrices(box_->state_.pose.translation(),
                                    Vector3<double>{2, 2, 2},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(box_->state_.pose.rotation().matrix(),
                                    Matrix3<double>::Identity(),
                                    TOLERANCE));

        double dt = 1.0;
        box_->AdvanceOneTimeStep(dt);

        // After the time step, the box should move with velocity (1, 0, 0)
        ASSERT_TRUE(CompareMatrices(box_->state_.pose.translation(),
                                    Vector3<double>{3, 2, 2},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(box_->state_.pose.rotation().matrix(),
                                    Matrix3<double>::Identity(),
                                    TOLERANCE));
    }

    void TestCylinderAdvanceOneTimeStep() {
        math::RollPitchYaw moving_cylinder_rpw = {M_PI/2.0, 0.0, 0.0};
        // Check the current pose of the cylinder
        ASSERT_TRUE(CompareMatrices(moving_cylinder_->state_.pose.translation(),
                                    Vector3<double>{0, 2, 3},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(
                            moving_cylinder_->state_.pose.rotation().matrix(),
                            moving_cylinder_rpw.ToMatrix3ViaRotationMatrix(),
                            TOLERANCE));

        double dt = 1.0;
        moving_cylinder_->AdvanceOneTimeStep(dt);

        // After the time step, the cylinder should move with velocity (1, 0, 0)
        ASSERT_TRUE(CompareMatrices(moving_cylinder_->state_.pose.translation(),
                                    Vector3<double>{1, 2, 3},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(
                            moving_cylinder_->state_.pose.rotation().matrix(),
                            moving_cylinder_rpw.ToMatrix3ViaRotationMatrix(),
                            TOLERANCE));
    }

    void TestSphereAdvanceOneTimeStep() {
        // Check the current pose of the sphere
        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.translation(),
                                    Vector3<double>{0, 0, 0},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.rotation().matrix(),
                                    Matrix3<double>::Identity(),
                                    TOLERANCE));

        double dt = 1.0;
        sphere_->AdvanceOneTimeStep(dt);

        // After the time step, the sphere should have a different pose with
        // the same translation but a different rotation. In particular,
        // by the formula dR/dt = [ω]R, where w is the angular momentum matrix,
        // Since the angular momentum w is constant, the new rotation matrix
        // after a time step should be R_new = (I + [w])R,
        // [w] = [0   0   0
        //        0   0  -π
        //        0   π   0] in this case. After normalizing to the closest
        // rotation matrix, the new rotation matrix should be
        // R_new = [1           0           0
        //          0   1/(√1+π²)  -π/(√1+π²)
        //          0   π/(√1+π²)   1/(√1+π²)]
        Matrix3<double> R_new;
        R_new << 1.0,                      0.0,                       0.0,
                 0.0,  1.0/sqrt(1 + M_PI*M_PI), -M_PI/sqrt(1 + M_PI*M_PI),
                 0.0, M_PI/sqrt(1 + M_PI*M_PI),   1.0/sqrt(1 + M_PI*M_PI);
        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.translation(),
                                    Vector3<double>{0, 0, 0},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.rotation().matrix(),
                                    R_new, TOLERANCE));

        sphere_->AdvanceOneTimeStep(dt);
        // After another time step, the new unnormalized rotation matrix should
        // be:     (I + [w])[1           0           0
        //                   0   1/(√1+π²)  -π/(√1+π²)
        //                   0   π/(√1+π²)   1/(√1+π²)]
        // R_new = 1/(√[(1-π²)²+4π²])[1     0     0
        //                            0   1-π²  -2π
        //                            0    2π  1-π²]
        double scale = sqrt(((1-M_PI*M_PI)*(1-M_PI*M_PI)+4*M_PI*M_PI));
        R_new << 1.0,                   0.0,                   0.0,
                 0.0, (1.0-M_PI*M_PI)/scale,       -2.0*M_PI/scale,
                 0.0,        2.0*M_PI/scale, (1.0-M_PI*M_PI)/scale;

        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.translation(),
                                    Vector3<double>{0, 0, 0},
                                    TOLERANCE));
        ASSERT_TRUE(CompareMatrices(sphere_->state_.pose.rotation().matrix(),
                                    R_new, TOLERANCE));
    }

    void TestWallBC() {
        // Test the wall boundary condition using half_space_ defined in the
        // SetUp routine.
        // First consider a point outside the halfspace, no boundary condition
        // shall be enforced.
        Vector3<double> pos0 = {3.0, 3.0, 3.0};
        Vector3<double> vel0 = {1.0, -1.0, 0.8};
        half_space_->ApplyBoundaryCondition(pos0, &vel0);
        ASSERT_TRUE(CompareMatrices(vel0, Vector3<double>{1.0, -1.0, 0.8},
                                                                    TOLERANCE));

        // Next consider a point on the boundary half plane, boundary condition
        // shall not be enforced, since the velocity is in the normal direction
        Vector3<double> pos1 = {2.0, 2.0, 2.0};
        Vector3<double> vel1 = {10.0, 10.0, 10.0};
        half_space_->ApplyBoundaryCondition(pos1, &vel1);
        ASSERT_TRUE(CompareMatrices(vel1, Vector3<double>{10.0, 10.0, 10.0},
                                    TOLERANCE));

        // Next consider a point on the boundary half plane, boundary condition
        // shall be enforced, since the velocity is in the opposite normal
        // direction, so the new velocity should be zero.
        Vector3<double> pos2 = {2.0, 2.0, 2.0};
        Vector3<double> vel2 = {-10.0, -10.0, -10.0};
        half_space_->ApplyBoundaryCondition(pos2, &vel2);
        ASSERT_TRUE(CompareMatrices(vel2, Vector3<double>::Zero(), TOLERANCE));

        // Next consider a point on the boundary half plane. The velocity is in
        // not the normal direction, but the normal impulse and the consequent
        // friction is so large that only the tangential velocity is left after
        // hitting the boundary.
        Vector3<double> pos3 = {2.0, 2.0, 2.0};
        Vector3<double> vel3 = {-11.0, -10.0, -10.0};
        half_space_->ApplyBoundaryCondition(pos3, &vel3);
        ASSERT_TRUE(CompareMatrices(vel3, Vector3<double>::Zero(), TOLERANCE));

        // Finally test the frictional wall boundary condition enforcement using
        // a different velocity such that the friction will not be too large
        // The relative velocity of the point is v = (-2, -1, -1), and the
        // ourward normal direction is (1.0, 1.0, 1.0). By algebra:
        // vₙ = (v ⋅ n)n = 4/3*(1, 1, 1), vₜ = v - vₙ = (-2/3, 1/3, 1/3)
        // v_new = vₜ - μ‖vₙ‖t = (1-0.2sqrt(2))*(-2/3, 1/3, 1/3)
        Vector3<double> pos4 = {2.0, 2.0, 2.0};
        Vector3<double> vel4 = {-2.0, -1.0, -1.0};
        half_space_->ApplyBoundaryCondition(pos4, &vel4);
        ASSERT_TRUE(CompareMatrices(vel4,
                                (1-half_space_->friction_coeff_*2.0*sqrt(2))/3.0
                                *Vector3<double>{-2.0, 1.0, 1.0},
                                TOLERANCE));
    }

    void TestSphereBC() {
        // Test the wall boundary condition using sphere_ defined in the
        // SetUp routine.
        // First consider a point outside the sphere, no boundary condition
        // shall be enforced.
        Vector3<double> pos0 = {3.0, 3.0, 3.0};
        Vector3<double> vel0 = {1.0, -1.0, 0.8};
        sphere_->ApplyBoundaryCondition(pos0, &vel0);
        ASSERT_TRUE(CompareMatrices(vel0, Vector3<double>{1.0, -1.0, 0.8},
                                                                    TOLERANCE));

        // Next consider a point on the sphere, boundary condition
        // shall be enforced, the translational velocity of the sphere at this
        // point is given by ω×r = (0, -pi, pi)/sqrt(3)
        // Then the relative velocity of the point is v = (-1, 0, 0), and the
        // ourward normal direction is (1.0, 1.0, 1.0). By algebra:
        // vₙ = (v ⋅ n)n = -1/3*(1, 1, 1), vₜ = v - vₙ = (-2/3, 1/3, 1/3)
        // v_new = vₜ - μ‖vₙ‖t = (2/3, -1/3, -1/3) - 0.2*sqrt(3)/6*(2, -1, -1)
        // In physical frame, v_new = (0, -pi, pi)/sqrt(3) + v_new
        Vector3<double> pos1 = {1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)};
        Vector3<double> vel1 = {-1.0, -M_PI/sqrt(3), M_PI/sqrt(3)};
        sphere_->ApplyBoundaryCondition(pos1, &vel1);
        ASSERT_TRUE(CompareMatrices(vel1, (1.0-0.2*sqrt(2)/2)
                                         *Vector3<double>{-2.0/3.0,
                                                          1.0/3.0,
                                                          1.0/3.0}
                                         +Vector3<double>{0.0,
                                                         -M_PI/sqrt(3),
                                                         M_PI/sqrt(3)},
                                         TOLERANCE));
    }

    void TestMovingSphereBC() {
        // Test the wall boundary condition using moving_sphere_ defined in the
        // SetUp routine.
        // First consider a point outside the sphere, no boundary condition
        // shall be enforced.
        Vector3<double> pos0 = {3.0, 3.0, 3.0};
        Vector3<double> vel0 = {1.0, -1.0, 0.8};
        moving_sphere_->ApplyBoundaryCondition(pos0, &vel0);
        ASSERT_TRUE(CompareMatrices(vel0, Vector3<double>{1.0, -1.0, 0.8},
                                                                    TOLERANCE));

        // Next consider a point on the sphere, boundary condition
        // shall be enforced, the translational velocity of the sphere at this
        // point is given by ω×r = (0, -pi, pi)/sqrt(3)+(1, 0, 0)
        // Then the relative velocity of the point is v = (-1, 0, 0), and the
        // ourward normal direction is (1.0, 1.0, 1.0). By algebra:
        // vₙ = (v ⋅ n)n = -1/3*(1, 1, 1), vₜ = v - vₙ = (-2/3, 1/3, 1/3)
        // v_new = vₜ - μ‖vₙ‖t = (2/3, -1/3, -1/3) - 0.2*sqrt(3)/6*(2, -1, -1)
        // In physical frame, v_new = (1, -pi, pi)/sqrt(3) + v_new
        Vector3<double> pos1 = {1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)};
        Vector3<double> vel1 = {0.0, -M_PI/sqrt(3), M_PI/sqrt(3)};
        moving_sphere_->ApplyBoundaryCondition(pos1, &vel1);
        ASSERT_TRUE(CompareMatrices(vel1, (1.0-0.2*sqrt(2)/2)
                                         *Vector3<double>{-2.0/3.0,
                                                          1.0/3.0,
                                                          1.0/3.0}
                                         +Vector3<double>{1.0,
                                                         -M_PI/sqrt(3),
                                                         M_PI/sqrt(3)},
                                         TOLERANCE));
    }

    void TestMovingCylindrical() {
        // First consider a particle with zero initial velocity not in the
        // cylinder, no boundary condition should be enforced.
        Vector3<double> pos0 = {3.0, 2.5, 2.5};
        Vector3<double> vel0 = Vector3<double>::Zero();
        moving_cylinder_->ApplyBoundaryCondition(pos0, &vel0);
        ASSERT_TRUE(CompareMatrices(vel0, Vector3<double>::Zero(), TOLERANCE));

        // After the cylinder move by dt=2.5, the point will lie in the cylinder
        // In this case, the relative velocity of the grid point is (-1, 0, 0),
        // vₙ = (v ⋅ n)n = (-1/2, 0, -1/2), vₜ = v - vₙ = (1/2, 0, -1/2)
        // v_new = vₜ - μ‖vₙ‖t = (1/2, 0, -1/2) - 0.1*(1, 0, -1)/2
        // In physical frame, v_new = (1, 0, 0) + (-0.45, 0, -0.45)
        double dt = 2.5;
        moving_cylinder_->AdvanceOneTimeStep(dt);
        ASSERT_TRUE(CompareMatrices(moving_cylinder_->state_.pose.translation(),
                                    Vector3<double>{2.5, 2, 3},
                                    TOLERANCE));
        moving_cylinder_->ApplyBoundaryCondition(pos0, &vel0);
        ASSERT_TRUE(CompareMatrices(vel0, Vector3<double>{0.55, 0.0, -0.45},
                                    TOLERANCE));
    }

    std::unique_ptr<CollisionObject> half_space_;
    std::unique_ptr<CollisionObject> box_;
    std::unique_ptr<CollisionObject> sphere_;
    std::unique_ptr<CollisionObject> moving_cylinder_;
    std::unique_ptr<CollisionObject> moving_sphere_;
};

namespace {

TEST_F(CollisionObjectTest, TestAdvanceOneTimeStep) {
    TestBoxAdvanceOneTimeStep();
    TestCylinderAdvanceOneTimeStep();
    TestSphereAdvanceOneTimeStep();
}

TEST_F(CollisionObjectTest, TestApplyBC) {
    TestWallBC();
    TestSphereBC();
    TestMovingSphereBC();
    TestMovingCylindrical();
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
