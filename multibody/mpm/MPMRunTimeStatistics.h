#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

// Store the run time statistic of the MPM simulation
struct MPMRunTimeStatistics {
    MPMRunTimeStatistics(): time_total(0.0),
                            time_io(0.0),
                            time_update_stress_and_plasticity(0.0),
                            time_setup_transfer(0.0),
                            time_P2G(0.0),
                            time_update_grid_velocity(0.0),
                            time_apply_external_forces(0.0),
                            time_collision_objects_update(0.0),
                            time_enforce_bc(0.0),
                            time_G2P(0.0),
                            time_advect_particles(0.0) {}

    double time_total;
    double time_io;
    double time_update_stress_and_plasticity;
    double time_setup_transfer;
    double time_P2G;
    double time_update_grid_velocity;
    double time_apply_external_forces;
    double time_collision_objects_update;
    double time_enforce_bc;
    double time_G2P;
    double time_advect_particles;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

