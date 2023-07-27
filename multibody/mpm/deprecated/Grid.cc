#include "drake/multibody/mpm/Grid.h"

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

Vector3<double> Grid::get_position(int idx_flat) const {
    DRAKE_ASSERT(idx_flat >= 0 && idx_flat < num_gridpt_);
    Vector3<int> expanded = Expand1DIndex(idx_flat);
    return get_position(expanded(0), expanded(1), expanded(2));
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
}

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

TotalMassEnergyMomentum Grid::GetTotalMassAndMomentum() const {
    TotalMassEnergyMomentum sum_state;
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

void Grid::writeGrid2obj(const std::string& filename){
    std::ofstream myfile;
    myfile.open(filename);
    for (int idx_flat = 0; idx_flat < num_gridpt_; idx_flat++){
        std::cout << idx_flat << std::endl;
        Vector3<double> position_at = get_position(idx_flat);
        myfile << "v " + std::to_string(position_at(0)) + " " + std::to_string(position_at(1)) + " " + std::to_string(position_at(2)) +"\n";
    }
    myfile.close();
}

void Grid::writeGridVelocity2obj(const std::string& filename) {
    std::ofstream myfile;
    myfile.open(filename);
    for (int idx_flat = 0; idx_flat < num_gridpt_; idx_flat++){
        Vector3<double> position_at = get_position(idx_flat);
        Vector3<double> velocity_at = get_velocity(idx_flat);
        myfile << "v " + std::to_string(position_at(0)) + " " + std::to_string(position_at(1)) + " " + std::to_string(position_at(2)) +"\n";
        myfile << "v " + std::to_string(position_at(0)+velocity_at(0)) + " " + std::to_string(position_at(1)+velocity_at(1)) + " " + std::to_string(position_at(2)+velocity_at(2)) +"\n";
    }
    for (int idx_flat = 0; idx_flat < num_gridpt_; idx_flat++){
        myfile << "l " + std::to_string(idx_flat*2+1) + " " + std::to_string(idx_flat*2+2) + "\n";
    }
    myfile.close();
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
