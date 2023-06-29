#pragma once

#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/Grid.h"

namespace drake {
namespace multibody {
namespace mpm {

// A class providing the utitlities to apply gravitational forces to the grid
class GravitationalForce {
 public:
    GravitationalForce();
    explicit GravitationalForce(Vector3<double> g);

    // Apply the gravitational forces to grid points. (Equivalently, apply
    // gravitational acceleration to the velocity)
    // v_new = v_prev + dt*gravitational_acceleration
    void ApplyGravitationalForces(double dt, Grid* grid) const;

 private:
    Vector3<double> gravitational_acceleration_;
};  // class GravitationalForce

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
