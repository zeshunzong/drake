#include "drake/multibody/mpm/GravitationalForceDenseGrid.h"

namespace drake {
namespace multibody {
namespace mpm {

GravitationalForce::GravitationalForce():
                                gravitational_acceleration_(0.0, 0.0, -9.81) {}

GravitationalForce::GravitationalForce(Vector3<double> g):
                                gravitational_acceleration_(std::move(g)) {}

void GravitationalForce::ApplyGravitationalForces(double dt, Grid* grid)
                                                                        const {
    Vector3<double> dv = dt*gravitational_acceleration_;
    // Gravitational acceleration
    for (int i = 0; i < grid->get_num_gridpt(); ++i) {
        const Vector3<double>& velocity_i = grid->get_velocity(i);
        grid->set_velocity(i, velocity_i + dv);
    }
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
