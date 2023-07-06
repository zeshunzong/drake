#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/KinematicCollisionObjects.h"
#include "drake/multibody/mpm/Particles.h"
#include "drake/multibody/mpm/Utils.h"

namespace drake {
namespace multibody {
namespace mpm {

class SparseGrid {
 public:
    SparseGrid() = default;
    explicit SparseGrid(double h);

    void reserve(double capacity);
    int get_num_active_gridpt() const;
    double get_h() const;

    // For below (i, j, k), we expect users pass in the coordinate in the
    // index space as documented above.
    Vector3<double> get_position(const Vector3<int>& index_3d) const;
    const Vector3<double>& get_velocity(const Vector3<int>& index_3d) const;
    double get_mass(const Vector3<int>& index_3d) const;
    const Vector3<double>& get_force(const Vector3<int>& index_3d) const;
    
    const Vector3<double>& get_velocity(size_t index_1d) const;

    void set_velocity(const Vector3<int>& index_3d,
                      const Vector3<double>& velocity);
    void set_mass(const Vector3<int>& index_3d, double mass);
    void set_force(const Vector3<int>& index_3d, const Vector3<double>& force);

    void set_velocity(size_t index_1d, const Vector3<double>& velocity);

    // Accumulate the state at (i, j, k) with the given value
    void AccumulateVelocity(const Vector3<int>& index_3d,
                            const Vector3<double>& velocity);
    void AccumulateMass(const Vector3<int>& index_3d, double mass);
    void AccumulateForce(const Vector3<int>& index_3d,
                         const Vector3<double>& force);

    // Check if the given grid point is an active grid point
    bool is_active(const Vector3<int>& gridpt) const;

    // Update active_gridpts_ with the given input. Assume the input is
    // already sorted without any duplicates.
    void UpdateActiveGridPoints(const std::vector<Vector3<int>>& active_gridpts);

    // loop over all particles to mark active grid nodes, also sort them
    void UpdateActiveGridPoints(const std::vector<Vector3<int>>& batch_indices,
                                const Particles& particles);

    // Rescale the velocities_ vector by the mass_, used in P2G where we
    // temporarily store momentum mv into velocities
    void RescaleVelocities();

    // Set masses, velocities and forces to be 0
    void ResetStates();

    // Reduce an 3D (i, j, k) index in the index space to a corresponding
    // 1D index on the sparse grid
    size_t Reduce3DIndex(const Vector3<int>& index_3d) const;

    // Expand a linear 1D index on the sparse grid to a 3D (i, j, k) index
    Vector3<int> Expand1DIndex(size_t index_1d) const;

    // Assume the explicit grid force at step n f^n is calculated and stored in
    // the member variable, we update the velocity with the formula
    // v^{n+1} = v^n + dt*f^n/m^n
    void UpdateVelocity(double dt);

    // Enforce wall boundary conditions using the given kinematic collision
    // objects. The normal of the given collision object is the outward pointing
    // normal from the interior of the object to the exterior of the object. We
    // strongly impose this Dirichlet boundary conditions.
    std::tuple<double, double, double, double> EnforceBoundaryCondition(
                                    const KinematicCollisionObjects& objects, double dt, double t);

    // update_velocity takes in argument (position, time, *velocity) to
    // overwrite 'velocity' with the new velocity given position and time
    void OverwriteGridVelocity(std::function<void(Vector3<double>,double,
                                                  Vector3<double>*)>
                               update_velocity, double t);

    // Return the sum of mass, momentum and angular momentum of all grid points
    TotalMassEnergyMomentum GetTotalMassAndMomentum() const;

 private:
    int num_active_gridpts_;
    double h_;

    // A unordered map from the 3D physical index to the 1D index in the memory
    std::unordered_map<Vector3<int>, size_t> index_map_{};
    // Sorted active grid points' indices by lexiographical ordering
    std::vector<Vector3<int>> active_gridpts_{};
    std::vector<Vector3<double>> velocities_{};
    std::vector<double> masses_{};
    std::vector<Vector3<double>> forces_{};
};  // class Grid

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
