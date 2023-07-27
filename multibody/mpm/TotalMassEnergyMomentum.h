#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

// Store the sum of mass, momentum and angular momentum of the grid/particles
// The angular momentum is about the origin in the world frame
// TODO(yiminlin.tri): change name and doc to include energy
template <typename T>
struct TotalMassEnergyMomentum {
    T sum_mass;
    T sum_kinetic_energy;
    T sum_strain_energy;
    T sum_potential_energy;
    Vector3<T> sum_momentum;
    Vector3<T> sum_angular_momentum;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
