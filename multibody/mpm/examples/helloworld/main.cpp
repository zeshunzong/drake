#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>


#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
// #include "drake/common/filesystem.h"
// #include "drake/common/temp_directory.h"
// #include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/mpm/Grid.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/TotalMassEnergyMomentum.h"
#include "drake/multibody/mpm/Particles.h"
#include "drake/multibody/mpm/MPMParameters.h"

namespace drake {
namespace multibody {
namespace mpm {

void velocity_field(Vector3<double>, double, Vector3<double>*) { }

int DoMain() {
    double CFL = 0.01;
    std::cout << CFL << std::endl;

    Vector3<int> bottomcorner = {0,0,0};
    Vector3<int> num_gridpt_1D = {2,3,4};

    Grid g(num_gridpt_1D, 0.1, bottomcorner);
    std::cout << g.get_num_gridpt() << std::endl;

    g.writeGrid2obj("grid_position.obj");
    g.writeGridVelocity2obj("grid_velocity.obj");
    // MPMParameters::PhysicalParameters p_param {
    //     {0.0, 0.0, -9.81},                      // Gravitational acceleration
    //     velocity_field,
    // };

    // MPMParameters::SolverParameters s_param {
    //     1e-0,                                  // End time
    //     5e-4,                                  // Time step size
    //     0.05,                                   // Grid size
    //     CFL,                                  // CFL
    // };

    // MPMParameters::IOParameters io_param {
    //     "mpm-test",                            // case name
    //     "/home/yiminlin/Desktop/drake/multibody/fem/mpm-dev/examples/advecting-ball/outputs-CFL.01",       // output directory name
    //     0.04,                                  // Interval of outputting
    // };

    // MPMParameters param {p_param, s_param, io_param};
    // auto driver = std::make_unique<MPMDriver>(std::move(param));

    // KinematicCollisionObjects objects = KinematicCollisionObjects();

    // // Initialize a sphere
    // double radius = 0.2;
    // SphereLevelSet level_set_sphere = SphereLevelSet(radius);
    // Vector3<double> translation_sphere = {0.0, 0.0, 0.0};
    // math::RigidTransform<double> pose_sphere =
    //                         math::RigidTransform<double>(translation_sphere);
    // multibody::SpatialVelocity<double> velocity_sphere;
    // velocity_sphere.translational() = Vector3<double>::Zero();
    // velocity_sphere.rotational() = Vector3<double>{0.0, 0.0, 0.0};

    // double E = 8e4;
    // double nu = 0.49;
    // std::unique_ptr<CorotatedElasticModel> elastoplastic_model
    //         = std::make_unique<CorotatedElasticModel>(E, nu);
    // MPMDriver::MaterialParameters m_param_sphere{
    //                                             std::move(elastoplastic_model),
    //                                             1200,
    //                                             velocity_sphere,
    //                                             1
    //                                             };

    // driver->InitializeKinematicCollisionObjects(std::move(objects));
    // driver->InitializeParticles(level_set_sphere, pose_sphere,
    //                             std::move(m_param_sphere));
    // driver->DoTimeStepping();

    return 0;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main() {
    return drake::multibody::mpm::DoMain();
}
