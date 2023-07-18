#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Partio.h>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
// #include "drake/common/filesystem.h"
#include "drake/common/temp_directory.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/MPMDriver.h"
#include "drake/multibody/mpm/SpatialVelocityTimeDependent.h"
#include "drake/multibody/math/spatial_velocity.h"

namespace drake {
namespace multibody {
namespace mpm {

void velocity_field(Vector3<double>, double, Vector3<double>*) { }

int DoMain() {
    // Vector3<double> gravity = {0.0, 0.0, -9.81};
    Vector3<double> gravity = {0.0, 0.0, 0.0};
    multibody::SpatialVelocity<double> velocity_sphere;
    velocity_sphere.translational() = Vector3<double>{5.0, 0.0, 0.0};
    velocity_sphere.rotational() = Vector3<double>{0.0, 0.0, 0.0};


    MPMParameters::PhysicalParameters p_param {
        gravity,                      // Gravitational acceleration
        velocity_field,
    };

    MPMParameters::SolverParameters s_param {
        2e-0,                                  // End time
        5e-4,                                  // Time step size
        0.02,                                   // Grid size
        0.75,                                  // CFL
    };

    MPMParameters::IOParameters io_param {
        "mpm-test",                            // case name
        "/home/zeshunzong/Documents/drake/multibody/mpm/examples/ball_on_floor/outputs",       // output directory name
        0.04,                                  // Interval of outputting
    };

    MPMParameters param {p_param, s_param, io_param};
    auto driver = std::make_unique<MPMDriver>(std::move(param));

    KinematicCollisionObjects objects = KinematicCollisionObjects();

    // Initialize the left wall
    SpatialVelocity<double> zero_velocity;
    zero_velocity.SetZero();
    std::unique_ptr<SpatialVelocityTimeDependent> left_hand_velocity_ptr =
        std::make_unique<SpatialVelocityTimeDependent>(zero_velocity);
    double left_hand_mu = 1.0;
    Vector3<double> left_hand_xscale = {0.2, 100.0, 100.0};
    std::unique_ptr<AnalyticLevelSet> left_hand_level_set =
                            std::make_unique<BoxLevelSet>(left_hand_xscale);
    Vector3<double> left_hand_translation = {-1.0, 0.0, 0.0};
    math::RigidTransform<double> left_hand_pose =
                            math::RigidTransform<double>(left_hand_translation);
    objects.AddCollisionObject(std::move(left_hand_level_set), std::move(left_hand_pose),
                               std::move(left_hand_velocity_ptr), left_hand_mu);

    // Initialize the right wall
    std::unique_ptr<SpatialVelocityTimeDependent> right_hand_velocity_ptr =
        std::make_unique<SpatialVelocityTimeDependent>(zero_velocity);
    double right_hand_mu = 1.0;
    Vector3<double> right_hand_xscale = {0.2, 100.0, 100.0};
    std::unique_ptr<AnalyticLevelSet> right_hand_level_set =
                            std::make_unique<BoxLevelSet>(right_hand_xscale);
    Vector3<double> right_hand_translation = {1.0, 0.0, 0.0};
    math::RigidTransform<double> right_hand_pose =
                            math::RigidTransform<double>(right_hand_translation);
    objects.AddCollisionObject(std::move(right_hand_level_set), std::move(right_hand_pose),
                               std::move(right_hand_velocity_ptr), right_hand_mu);

    // Initialize a sphere
    double radius = 0.2;
    SphereLevelSet level_set_sphere = SphereLevelSet(radius);
    Vector3<double> translation_sphere = {0.0, 0.0, 0.0};
    math::RigidTransform<double> pose_sphere =
                            math::RigidTransform<double>(translation_sphere);

    double E = 8e4;
    double nu = 0.4;
    std::unique_ptr<CorotatedElasticModel> elastoplastic_model
            = std::make_unique<CorotatedElasticModel>(E, nu);
    MPMDriver::MaterialParameters m_param_sphere{
                                                std::move(elastoplastic_model),
                                                500,
                                                velocity_sphere,
                                                1
                                                };

    driver->InitializeKinematicCollisionObjects(std::move(objects));
    driver->InitializeParticles(level_set_sphere, pose_sphere,
                                std::move(m_param_sphere));
    driver->DoTimeStepping();

    return 0;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main() {
    return drake::multibody::mpm::DoMain();
}
