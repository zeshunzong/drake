#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

// Store the sum of mass, momentum and angular momentum of the grid/particles
// The angular momentum is about the origin in the world frame
// TODO(yiminlin.tri): change name and doc to include energy
struct TotalMassEnergyMomentum {
    double sum_mass;
    double sum_kinetic_energy;
    double sum_strain_energy;
    double sum_potential_energy;
    Vector3<double> sum_momentum;
    Vector3<double> sum_angular_momentum;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
