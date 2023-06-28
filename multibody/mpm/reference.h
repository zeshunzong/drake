#include "drake/multibody/fem/mpm-dev/Grid.h"

namespace drake {
namespace multibody {
namespace mpm {

Grid::Grid(const Vector3<int>& num_gridpt_1D, double h,
           const Vector3<int>& bottom_corner):
           num_gridpt_(num_gridpt_1D(0)*num_gridpt_1D(1)*num_gridpt_1D(2)),
           num_gridpt_1D_(num_gridpt_1D), h_(h),
           bottom_corner_(bottom_corner) {
    int idx;
    DRAKE_ASSERT(num_gridpt_1D_(0) >= 0);
    DRAKE_ASSERT(num_gridpt_1D_(1) >= 0);
    DRAKE_ASSERT(num_gridpt_1D_(2) >= 0);
    DRAKE_ASSERT(h_ > 0.0);

    indices_ = std::vector<std::pair<int, Vector3<int>>>(num_gridpt_);
    velocities_ = std::vector<Vector3<double>>(num_gridpt_);
    masses_ = std::vector<double>(num_gridpt_);
    forces_ = std::vector<Vector3<double>>(num_gridpt_);

    // Initialize the positions of grid points
    for (int k = bottom_corner_(2);
             k < bottom_corner_(2) + num_gridpt_1D_(2); ++k) {
    for (int j = bottom_corner_(1);
             j < bottom_corner_(1) + num_gridpt_1D_(1); ++j) {
    for (int i = bottom_corner_(0);
             i < bottom_corner_(0) + num_gridpt_1D_(0); ++i) {
        idx = Reduce3DIndex(i, j, k);
        indices_[idx] = std::pair<int, Vector3<int>>(idx, {i, j, k});
    }
    }
    }
}

int Grid::get_num_gridpt() const {
    return num_gridpt_;
}

const Vector3<int>& Grid::get_num_gridpt_1D() const {
    return num_gridpt_1D_;
}

double Grid::get_h() const {
    return h_;
}

const Vector3<int>& Grid::get_bottom_corner() const {
    return bottom_corner_;
}

Vector3<double> Grid::get_position(int i, int j, int k) const {
    DRAKE_ASSERT(in_index_range(i, j, k));
    return Vector3<double>(i*h_, j*h_, k*h_);
}

const Vector3<double>& Grid::get_velocity(int i, int j, int k) const {
    DRAKE_ASSERT(in_index_range(i, j, k));
    return velocities_[Reduce3DIndex(i, j, k)];
}

double Grid::get_mass(int i, int j, int k) const {
    DRAKE_ASSERT(in_index_range(i, j, k));
    return masses_[Reduce3DIndex(i, j, k)];
}

const Vector3<double>& Grid::get_force(int i, int j, int k) const {
    DRAKE_ASSERT(in_index_range(i, j, k));
    return forces_[Reduce3DIndex(i, j, k)];
}

const Vector3<double>& Grid::get_velocity(int idx_flat) const {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    return velocities_[idx_flat];
}

double Grid::get_mass(int idx_flat) const {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    return masses_[idx_flat];
}

const Vector3<double>& Grid::get_force(int idx_flat) const {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    return forces_[idx_flat];
}

void Grid::set_velocity(int i, int j, int k, const Vector3<double>& velocity) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    velocities_[Reduce3DIndex(i, j, k)] = velocity;
}

void Grid::set_mass(int i, int j, int k, double mass) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    masses_[Reduce3DIndex(i, j, k)] = mass;
}

void Grid::set_force(int i, int j, int k, const Vector3<double>& force) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    forces_[Reduce3DIndex(i, j, k)] = force;
}

void Grid::set_velocity(int idx_flat, const Vector3<double>& velocity) {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    velocities_[idx_flat] = velocity;
}

void Grid::set_mass(int idx_flat, double mass) {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    masses_[idx_flat] = mass;
}

void Grid::set_force(int idx_flat, const Vector3<double>& force) {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    forces_[idx_flat] = force;
}

void Grid::AccumulateVelocity(int i, int j, int k,
                                        const Vector3<double>& velocity) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    velocities_[Reduce3DIndex(i, j, k)] += velocity;
}

void Grid::AccumulateMass(int i, int j, int k, double mass) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    masses_[Reduce3DIndex(i, j, k)] += mass;
}

void Grid::AccumulateForce(int i, int j, int k, const Vector3<double>& force) {
    DRAKE_ASSERT(in_index_range(i, j, k));
    forces_[Reduce3DIndex(i, j, k)] += force;
}

void Grid::AccumulateVelocity(const Vector3<int>& index_3d,
                                        const Vector3<double>& velocity) {
    AccumulateVelocity(index_3d(0), index_3d(1), index_3d(2), velocity);
}

void Grid::AccumulateMass(const Vector3<int>& index_3d, double mass) {
    AccumulateMass(index_3d(0), index_3d(1), index_3d(2), mass);
}

void Grid::AccumulateForce(const Vector3<int>& index_3d,
                           const Vector3<double>& force) {
    AccumulateForce(index_3d(0), index_3d(1), index_3d(2), force);
}

void Grid::RescaleVelocities() {
    for (int i = 0; i < num_gridpt_; ++i) {
        velocities_[i] = velocities_[i] / masses_[i];
    }
}

void Grid::ResetStates() {
    std::fill(masses_.begin(), masses_.end(), 0.0);
    std::fill(velocities_.begin(), velocities_.end(), Vector3<double>::Zero());
    std::fill(forces_.begin(), forces_.end(), Vector3<double>::Zero());
}

int Grid::Reduce3DIndex(int i, int j, int k) const {
    DRAKE_ASSERT(in_index_range(i, j, k));
    return (k-bottom_corner_(2))*(num_gridpt_1D_(0)*num_gridpt_1D_(1))
         + (j-bottom_corner_(1))*num_gridpt_1D_(0)
         + (i-bottom_corner_(0));
}

int Grid::Reduce3DIndex(const Vector3<int>& index_3d) const {
    return Reduce3DIndex(index_3d(0), index_3d(1), index_3d(2));
}

Vector3<int> Grid::Expand1DIndex(int idx) const {
    return Vector3<int>(
            bottom_corner_(0) + idx % num_gridpt_1D_(0),
            bottom_corner_(1) + (idx / num_gridpt_1D_(0)) % num_gridpt_1D_(1),
            bottom_corner_(2) + idx / (num_gridpt_1D_(0)*num_gridpt_1D_(1)));
}

const std::vector<std::pair<int, Vector3<int>>>& Grid::get_indices() const {
    return indices_;
}

bool Grid::in_index_range(int i, int j, int k) const {
    return ((i < bottom_corner_(0) + num_gridpt_1D_(0)) &&
            (j < bottom_corner_(1) + num_gridpt_1D_(1)) &&
            (k < bottom_corner_(2) + num_gridpt_1D_(2)) &&
            (i >= bottom_corner_(0)) &&
            (j >= bottom_corner_(1)) &&
            (k >= bottom_corner_(2)));
}#pragma once

#include <limits>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/mpm-dev/KinematicCollisionObjects.h"
#include "drake/multibody/fem/mpm-dev/TotalMassAndMomentum.h"

namespace drake {
namespace multibody {
namespace mpm {

// A grid class holding vectors of grid points' state. The grid can be
// visually represented as:
//
//                 h
//               o - o - o - o - o - o
//               |   |   |   |   |   |
//           o - o - o - o - o - o - o
//           |   |   |   |   |   |   |
//       o - o - o - o - o - o - o - o
//       |   |   |   |   |   |   |   |
//       o - o - o - o - o - o - o - o
//       |   |   |   |   |   |   |
//       o - o - o - o - o - o - o                  z   y
//       |   |   |   |   |   |                      | /
//       o - o - o - o - o - o                       -- x
//  bottom_corner
//
// The grid can be uniquely represented as a bottom corner and the number of
// grid points in x, y and z directions.
// Since we assume an uniform grid, the distance between neighboring grid points
// are all h. The bottom_corner is the grid node with the smallest (x, y, z)
// values. We assume the bottom corner aligns with the "index space" (i, j, k)
// , which is an 3D integer space (Z \times Z \times Z). The actual physical
// coordinate of the index space is defined through the mapping (ih, jh, kh).
// Positions of grid points will be stored as physical space coordinates.
// The vector num_gridpt_1D stores the number of grid points along x, y, and z
// directions. Figure above is an example with num_gridpt_1D = (6, 3, 4).
// We stores all the data holding states as 1D vectors, with linear indexing
// following the lexiographical order in x, y, z directions
class Grid {
 public:
    Grid() = default;
    Grid(const Vector3<int>& num_gridpt_1D, double h,
         const Vector3<int>& bottom_corner);

    int get_num_gridpt() const;
    const Vector3<int>& get_num_gridpt_1D() const;
    double get_h() const;
    const Vector3<int>& get_bottom_corner() const;

    // For below (i, j, k), we expect users pass in the coordinate in the
    // index space as documented above.
    Vector3<double> get_position(int i, int j, int k) const;
    const Vector3<double>& get_velocity(int i, int j, int k) const;
    double get_mass(int i, int j, int k) const;
    const Vector3<double>& get_force(int i, int j, int k) const;
    const std::vector<std::pair<int, Vector3<int>>>& get_indices() const;

    const Vector3<double>& get_velocity(int idx_flat) const;
    double get_mass(int idx_flat) const;
    const Vector3<double>& get_force(int idx_flat) const;

    void set_velocity(int i, int j, int k, const Vector3<double>& velocity);
    void set_mass(int i, int j, int k, double mass);
    void set_force(int i, int j, int k, const Vector3<double>& force);

    void set_velocity(int idx_flat, const Vector3<double>& velocity);
    void set_mass(int idx_flat, double mass);
    void set_force(int idx_flat, const Vector3<double>& force);

    // Accumulate the state at (i, j, k) with the given value
    void AccumulateVelocity(int i, int j, int k,
                                            const Vector3<double>& velocity);
    void AccumulateMass(int i, int j, int k, double mass);
    void AccumulateForce(int i, int j, int k, const Vector3<double>& force);
    void AccumulateVelocity(const Vector3<int>& index_3d,
                                            const Vector3<double>& velocity);
    void AccumulateMass(const Vector3<int>& index_3d, double mass);
    void AccumulateForce(const Vector3<int>& index_3d,
                         const Vector3<double>& force);

    // Rescale the velocities_ vector by the mass_, used in P2G where we
    // temporarily store momentum mv into velocities
    void RescaleVelocities();

    // Set masses, velocities and forces to be 0
    void ResetStates();

    // Reduce an 3D (i, j, k) index in the index space to a corresponding
    // linear lexiographical ordered index
    int Reduce3DIndex(int i, int j, int k) const;
    int Reduce3DIndex(const Vector3<int>& index_3d) const;

    // Expand a linearly lexiographical ordered index to a 3D index (i, j, k)
    // in the index space
    Vector3<int> Expand1DIndex(int idx) const;

    // Check the passed in (i, j, k) lies within the index range of this Grid
    bool in_index_range(int i, int j, int k) const;
    bool in_index_range(const Vector3<int>& index_3d) const;

    // Assume the explicit grid force at step n f^n is calculated and stored in
    // the member variable, we update the velocity with the formula
    // v^{n+1} = v^n + dt*f^n/m^n
    void UpdateVelocity(double dt);

    // Enforce wall boundary conditions using the given kinematic collision
    // objects. The normal of the given collision object is the outward pointing
    // normal from the interior of the object to the exterior of the object. We
    // strongly impose this Dirichlet boundary conditions.
    void EnforceBoundaryCondition(const KinematicCollisionObjects& objects);

    // Return the sum of mass, momentum and angular momentum of all grid points
    TotalMassAndMomentum GetTotalMassAndMomentum() const;

 private:
    int num_gridpt_;
    Vector3<int> num_gridpt_1D_;              // Number of grid points on the
                                              // grid along x, y, z directions
    double h_;
    Vector3<int> bottom_corner_{};

    // The vector of 1D and 3D indices of grid points, ordered lexiographically
    std::vector<std::pair<int, Vector3<int>>> indices_{};
    std::vector<Vector3<double>> velocities_{};
    std::vector<double> masses_{};
    std::vector<Vector3<double>> forces_{};
};  // class Grid

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

#pragma once

#include <limits>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/mpm-dev/KinematicCollisionObjects.h"
#include "drake/multibody/fem/mpm-dev/TotalMassAndMomentum.h"

namespace drake {
namespace multibody {
namespace mpm {

// A grid class holding vectors of grid points' state. The grid can be
// visually represented as:
//
//                 h
//               o - o - o - o - o - o
//               |   |   |   |   |   |
//           o - o - o - o - o - o - o
//           |   |   |   |   |   |   |
//       o - o - o - o - o - o - o - o
//       |   |   |   |   |   |   |   |
//       o - o - o - o - o - o - o - o
//       |   |   |   |   |   |   |
//       o - o - o - o - o - o - o                  z   y
//       |   |   |   |   |   |                      | /
//       o - o - o - o - o - o                       -- x
//  bottom_corner
//
// The grid can be uniquely represented as a bottom corner and the number of
// grid points in x, y and z directions.
// Since we assume an uniform grid, the distance between neighboring grid points
// are all h. The bottom_corner is the grid node with the smallest (x, y, z)
// values. We assume the bottom corner aligns with the "index space" (i, j, k)
// , which is an 3D integer space (Z \times Z \times Z). The actual physical
// coordinate of the index space is defined through the mapping (ih, jh, kh).
// Positions of grid points will be stored as physical space coordinates.
// The vector num_gridpt_1D stores the number of grid points along x, y, and z
// directions. Figure above is an example with num_gridpt_1D = (6, 3, 4).
// We stores all the data holding states as 1D vectors, with linear indexing
// following the lexiographical order in x, y, z directions
class Grid {
 public:
    Grid() = default;
    Grid(const Vector3<int>& num_gridpt_1D, double h,
         const Vector3<int>& bottom_corner);

    int get_num_gridpt() const;
    const Vector3<int>& get_num_gridpt_1D() const;
    double get_h() const;
    const Vector3<int>& get_bottom_corner() const;

    // For below (i, j, k), we expect users pass in the coordinate in the
    // index space as documented above.
    Vector3<double> get_position(int i, int j, int k) const;
    const Vector3<double>& get_velocity(int i, int j, int k) const;
    double get_mass(int i, int j, int k) const;
    const Vector3<double>& get_force(int i, int j, int k) const;
    const std::vector<std::pair<int, Vector3<int>>>& get_indices() const;

    const Vector3<double>& get_velocity(int idx_flat) const;
    double get_mass(int idx_flat) const;
    const Vector3<double>& get_force(int idx_flat) const;

    void set_velocity(int i, int j, int k, const Vector3<double>& velocity);
    void set_mass(int i, int j, int k, double mass);
    void set_force(int i, int j, int k, const Vector3<double>& force);

    void set_velocity(int idx_flat, const Vector3<double>& velocity);
    void set_mass(int idx_flat, double mass);
    void set_force(int idx_flat, const Vector3<double>& force);

    // Accumulate the state at (i, j, k) with the given value
    void AccumulateVelocity(int i, int j, int k,
                                            const Vector3<double>& velocity);
    void AccumulateMass(int i, int j, int k, double mass);
    void AccumulateForce(int i, int j, int k, const Vector3<double>& force);
    void AccumulateVelocity(const Vector3<int>& index_3d,
                                            const Vector3<double>& velocity);
    void AccumulateMass(const Vector3<int>& index_3d, double mass);
    void AccumulateForce(const Vector3<int>& index_3d,
                         const Vector3<double>& force);

    // Rescale the velocities_ vector by the mass_, used in P2G where we
    // temporarily store momentum mv into velocities
    void RescaleVelocities();

    // Set masses, velocities and forces to be 0
    void ResetStates();

    // Reduce an 3D (i, j, k) index in the index space to a corresponding
    // linear lexiographical ordered index
    int Reduce3DIndex(int i, int j, int k) const;
    int Reduce3DIndex(const Vector3<int>& index_3d) const;

    // Expand a linearly lexiographical ordered index to a 3D index (i, j, k)
    // in the index space
    Vector3<int> Expand1DIndex(int idx) const;

    // Check the passed in (i, j, k) lies within the index range of this Grid
    bool in_index_range(int i, int j, int k) const;
    bool in_index_range(const Vector3<int>& index_3d) const;

    // Assume the explicit grid force at step n f^n is calculated and stored in
    // the member variable, we update the velocity with the formula
    // v^{n+1} = v^n + dt*f^n/m^n
    void UpdateVelocity(double dt);

    // Enforce wall boundary conditions using the given kinematic collision
    // objects. The normal of the given collision object is the outward pointing
    // normal from the interior of the object to the exterior of the object. We
    // strongly impose this Dirichlet boundary conditions.
    void EnforceBoundaryCondition(const KinematicCollisionObjects& objects);

    // Return the sum of mass, momentum and angular momentum of all grid points
    TotalMassAndMomentum GetTotalMassAndMomentum() const;

 private:
    int num_gridpt_;
    Vector3<int> num_gridpt_1D_;              // Number of grid points on the
                                              // grid along x, y, z directions
    double h_;
    Vector3<int> bottom_corner_{};

    // The vector of 1D and 3D indices of grid points, ordered lexiographically
    std::vector<std::pair<int, Vector3<int>>> indices_{};
    std::vector<Vector3<double>> velocities_{};
    std::vector<double> masses_{};
    std::vector<Vector3<double>> forces_{};
};  // class Grid

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

bool Grid::in_index_range(const Vector3<int>& index_3d) const {
    return in_index_range(index_3d(0), index_3d(1), index_3d(2));
}

void Grid::UpdateVelocity(double dt) {
    for (int i = 0; i < num_gridpt_; ++i) {
        // Only update at grid points with nonzero masses
        if (masses_[i] > 0.0) {
            velocities_[i] += dt*forces_[i]/masses_[i];
        }
    }
}

void Grid::EnforceBoundaryCondition(const KinematicCollisionObjects& objects) {
    // For all grid points, enforce frictional wall boundary condition
    for (const auto& [index_flat, index_3d] : indices_) {
        // Only enforce at grid points with nonzero masses
        if (masses_[index_flat] > 0.0) {
            objects.ApplyBoundaryConditions(get_position(index_3d(0),
                                                         index_3d(1),
                                                         index_3d(2)),
                                            &velocities_[index_flat]);
        }
    }
}

TotalMassAndMomentum Grid::GetTotalMassAndMomentum() const {
    TotalMassAndMomentum sum_state;
    sum_state.sum_mass             = 0.0;
    sum_state.sum_momentum         = Vector3<double>::Zero();
    sum_state.sum_angular_momentum = Vector3<double>::Zero();
    for (int k = bottom_corner_(2);
                k < bottom_corner_(2)+num_gridpt_1D_(2); ++k) {
    for (int j = bottom_corner_(1);
                j < bottom_corner_(1)+num_gridpt_1D_(1); ++j) {
    for (int i = bottom_corner_(0);
                i < bottom_corner_(0)+num_gridpt_1D_(0); ++i) {
        double mi = get_mass(i, j, k);
        const Vector3<double> vi = get_velocity(i, j, k);
        const Vector3<double> xi = get_position(i, j, k);
        sum_state.sum_mass             += mi;
        sum_state.sum_momentum         += mi*vi;
        sum_state.sum_angular_momentum += mi*xi.cross(vi);
    }
    }
    }
    return sum_state;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
