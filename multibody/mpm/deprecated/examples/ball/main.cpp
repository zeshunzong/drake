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
#include "drake/common/temp_directory.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/MPMDriver.h"
#include "drake/multibody/math/spatial_velocity.h"

namespace drake {
namespace multibody {
namespace mpm {

// this is a function
void velocity_field(Vector3<double>, double, Vector3<double>*) { }

int DoMain() {
    double CFL = .2;
    multibody::SpatialVelocity<double> velocity_sphere;
    velocity_sphere.translational() = Vector3<double>{0.1, 0.1, 0.1};
    velocity_sphere.rotational() = Vector3<double>{M_PI/2, M_PI/2, M_PI/2};

    MPMParameters::PhysicalParameters p_param {
        {0.0, 0.0, 0.0},                      // Gravitational acceleration
        velocity_field,
    };

    MPMParameters::SolverParameters s_param {
        2e-0,                                  // End time
        4e-4,                                  // Time step size
        0.025,                                   // Grid size
        CFL,                                  // CFL
    };

    MPMParameters::IOParameters io_param {
        "mpm-test",                            // case name
        "/home/zeshunzong/Documents/drake/multibody/mpm/examples/ball/outputs",       // output directory name
        0.04,                                  // Interval of outputting
    };

    MPMParameters param {p_param, s_param, io_param};
    auto driver = std::make_unique<MPMDriver>(std::move(param)); // sparseGrid also initialized here

    // KinematicCollisionObjects objects = KinematicCollisionObjects();

    // Initialize a sphere
    double radius = 0.2;
    SphereLevelSet level_set_sphere = SphereLevelSet(radius);
    Vector3<double> translation_sphere = {0.0, 0.0, 0.0};
    math::RigidTransform<double> pose_sphere =
                            math::RigidTransform<double>(translation_sphere);

    double E = 5e4;
    double nu = 0.4;
    std::unique_ptr<CorotatedElasticModel> elastoplastic_model
            = std::make_unique<CorotatedElasticModel>(E, nu);
    MPMDriver::MaterialParameters m_param_sphere{
                                                std::move(elastoplastic_model),
                                                1000,
                                                velocity_sphere,
                                                1
                                                };

    // driver->InitializeKinematicCollisionObjects(std::move(objects));
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