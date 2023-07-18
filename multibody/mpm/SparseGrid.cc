#include "drake/multibody/mpm/SparseGrid.h"
#include <iostream>
namespace drake {
namespace multibody {
namespace mpm {

SparseGrid::SparseGrid(double h): num_active_gridpts_(0), h_(h),
                                  active_gridpts_(0) {
    std::cout << "create grid with h = " << h << std::endl;
                                  }

void SparseGrid::reserve(double capacity) {
    index_map_.reserve(capacity);
    active_gridpts_.reserve(capacity);
    velocities_.reserve(capacity);
    masses_.reserve(capacity);
    forces_.reserve(capacity);
}

int SparseGrid::get_num_active_gridpt() const {
    return num_active_gridpts_;
}

double SparseGrid::get_h() const {
    return h_;
}

Vector3<double> SparseGrid::get_position(const Vector3<int>& index_3d) const {
    return Vector3<double>(index_3d(0)*h_, index_3d(1)*h_, index_3d(2)*h_);
}

const Vector3<double>& SparseGrid::get_velocity(const Vector3<int>& index_3d)
                                                                        const {
    return velocities_[Reduce3DIndex(index_3d)];
}

double SparseGrid::get_mass(const Vector3<int>& index_3d) const {
    return masses_[Reduce3DIndex(index_3d)];
}

const Vector3<double>& SparseGrid::get_force(const Vector3<int>& index_3d)
                                                                        const {
    return forces_[Reduce3DIndex(index_3d)];
}

const Vector3<double>& SparseGrid::get_velocity(size_t index_1d) const {
    return velocities_[index_1d];
}

void SparseGrid::set_velocity(const Vector3<int>& index_3d,
                        const Vector3<double>& velocity) {
    velocities_[Reduce3DIndex(index_3d)] = velocity;
}

void SparseGrid::set_mass(const Vector3<int>& index_3d, double mass) {
    masses_[Reduce3DIndex(index_3d)] = mass;
}

void SparseGrid::set_force(const Vector3<int>& index_3d,
                     const Vector3<double>& force) {
    forces_[Reduce3DIndex(index_3d)] = force;
}

void SparseGrid::set_velocity(size_t index_1d,
                              const Vector3<double>& velocity) {
    velocities_[index_1d] = velocity;
}

void SparseGrid::AccumulateVelocity(const Vector3<int>& index_3d,
                                    const Vector3<double>& velocity) {
    velocities_[Reduce3DIndex(index_3d)] += velocity;
}

void SparseGrid::AccumulateMass(const Vector3<int>& index_3d, double mass) {
    masses_[Reduce3DIndex(index_3d)] += mass;
}

void SparseGrid::AccumulateForce(const Vector3<int>& index_3d,
                                 const Vector3<double>& force) {
    forces_[Reduce3DIndex(index_3d)] += force;
}

bool SparseGrid::is_active(const Vector3<int>& gridpt) const {
    return index_map_.count(gridpt) == 1;
}

void SparseGrid::UpdateActiveGridPoints(
                            const std::vector<Vector3<int>>& active_gridpts) {
    num_active_gridpts_ = active_gridpts.size();
    for (int i = 0; i < num_active_gridpts_; ++i) {
        active_gridpts_[i] = active_gridpts[i];
        index_map_[active_gridpts_[i]] = i;
    }
}

void SparseGrid::UpdateActiveGridPoints(const std::vector<Vector3<int>>& batch_indices,
                                        const Particles& particles) {
    // Clear active grid points
    num_active_gridpts_ = 0;
    active_gridpts_.clear();
    index_map_.clear();
    // Determine the set of active grids points by iterating through all
    // particles
    for (int p = 0; p < particles.get_num_particles(); ++p) {
        const Vector3<int>& batch_idx_3D = batch_indices[p];
        for (int c = -1; c <= 1; ++c) {
        for (int b = -1; b <= 1; ++b) {
        for (int a = -1; a <= 1; ++a) {
            Vector3<int> gridpt = {batch_idx_3D(0)+a, batch_idx_3D(1)+b,
                                   batch_idx_3D(2)+c}; // get all 27 neighboring grid nodes to particle p
            if (index_map_.count(gridpt) == 0) { // if this node is not active, mark as active here
                active_gridpts_[num_active_gridpts_++] = gridpt;
                index_map_[gridpt] = 1;
            }
        }
        }
        }
    }

    std::sort(active_gridpts_.begin(), active_gridpts_.end(),
              [](Vector3<int> pt0, Vector3<int> pt1) {
                return ((pt0(2) < pt1(2)) ||
                        ((pt0(2) == pt1(2)) && (pt0(1) < pt1(1))) ||
                        ((pt0(2) == pt1(2)) && (pt0(1) == pt1(1))
                      && (pt0(0) < pt1(0))));
              });

    for (int i = 0; i < num_active_gridpts_; ++i) {
        index_map_[active_gridpts_[i]] = i;
    }
}

void SparseGrid::RescaleVelocities() {
    for (int i = 0; i < num_active_gridpts_; ++i) {
        velocities_[i] = velocities_[i] / masses_[i];
    }
}

void SparseGrid::ResetStates() {
    std::fill(masses_.begin(), masses_.begin()+num_active_gridpts_, 0.0);
    std::fill(velocities_.begin(), velocities_.begin()+num_active_gridpts_,
              Vector3<double>::Zero());
    std::fill(forces_.begin(), forces_.begin()+num_active_gridpts_,
              Vector3<double>::Zero());
}

size_t SparseGrid::Reduce3DIndex(const Vector3<int>& index_3d) const {
    return index_map_.at(index_3d);
}

Vector3<int> SparseGrid::Expand1DIndex(size_t index_1D) const {
    return active_gridpts_[index_1D];
}

void SparseGrid::UpdateVelocity(double dt) {
    for (int i = 0; i < num_active_gridpts_; ++i) {
        velocities_[i] += dt*forces_[i]/masses_[i];
    }
}

std::tuple<double, double, double, double> SparseGrid::EnforceBoundaryCondition(
                                    const KinematicCollisionObjects& objects,
                                    double dt, double t) {
    // For all grid points, enforce frictional wall boundary condition
    double impulse_n = 0.0;
    double impulse_t = 0.0;
    double impulse_gravity_n = 0.0;
    double impulse_gravity_t = 0.0;
    // TODO(yiminlin.tri): hardcode normal and gravity .............
    // // For the slope case:
    // double angle = M_PI/12;
    // Vector3<double> wall_normal = {sin(angle), 0.0, cos(angle)};

    // For the shaking sphere case:
    Vector3<double> wall_normal = {1.0, 0.0, 0.0};

    // // For the stick slip transition:
    // Vector3<double> wall_normal = {0.0, 0.0, 1.0};

    // Vector3<double> gravity = {0.0, 0.0, -9.81};
    // double g_n_norm = gravity.dot(wall_normal);
    // Vector3<double> g_n = g_n_norm*wall_normal;
    // Vector3<double> g_t = gravity - g_n;
    // double g_t_norm = g_t.norm();
    // double g_n_norm_dt = g_n_norm*dt;
    // double g_t_norm_dt = g_t_norm*dt;

    double sum_mass = 0.0;
    for (int i = 0; i < num_active_gridpts_; ++i) {
        double mi = masses_[i];
        sum_mass += mi;
        Vector3<double> prev_v = velocities_[i];
        Vector3<double> xi = get_position(active_gridpts_[i]);
        objects.ApplyBoundaryConditions(xi, &velocities_[i]);
        Vector3<double> dv = velocities_[i] - prev_v;
        double dv_n_norm = dv.dot(wall_normal);
        Vector3<double> dv_n = dv_n_norm*wall_normal;
        Vector3<double> dv_t = dv - dv_n;
        // double dv_t_norm = dv_t.norm();
        // // For others
        // impulse_n += mi*dv_n_norm;
        // impulse_t += mi*dv_t_norm;
        // impulse_gravity_n += mi*g_n_norm_dt;
        // impulse_gravity_t += mi*g_t_norm_dt;

        // For the shaking sphere case
        if (xi[0] < 0.0 && xi[2] >= 0.001) {
            impulse_n += mi*dv_n_norm;
            // impulse_t += mi*dv_t_norm;
            impulse_t += mi*dv_t[2];
            // impulse_t += mi*dv_t[2];
        }
    }

    double freq = 0.4; // multiple of 0.04
    static bool has_down;
    static bool has_up;
    static bool has_move_up;
    if (t == dt) {
        has_down = false;
        has_up = false;
        has_move_up = false;
    }
    if (t >= 1.36) {
        if (static_cast<int>(trunc((t-1.36) / freq)) % 2 == 0) {
            if (!has_down) {
                impulse_gravity_t += 2.0*sum_mass;
                has_down = true;
                has_up = false;
            }
        } else {
            if (!has_up) {
                impulse_gravity_t -= 2.0*sum_mass;
                has_up = true;
                has_down = false;
            }
        }
    }
    // Move upward
    if (t > 0.36 && !has_move_up) {
        impulse_gravity_t += sum_mass*dt;
        has_move_up = true;
    }
    if (t > 0.36) {
        // impulse_gravity_n += sum_mass*g_n_norm_dt;
        impulse_gravity_t -= -9.81*sum_mass*dt;
    }

    return std::tuple<double, double, double, double>(impulse_n, impulse_t, impulse_gravity_n, impulse_gravity_t);
}

void SparseGrid::OverwriteGridVelocity(std::function<void(Vector3<double>,
                                                          double,
                                                          Vector3<double>*)>
                                       update_velocity, double t) {
    // For all grid points, overwrite grid velocity
    for (int i = 0; i < num_active_gridpts_; ++i) {
        update_velocity(get_position(active_gridpts_[i]), t, &velocities_[i]);
    }
}

TotalMassEnergyMomentum SparseGrid::GetTotalMassAndMomentum() const {
    TotalMassEnergyMomentum sum_state;
    sum_state.sum_mass             = 0.0;
    sum_state.sum_momentum         = Vector3<double>::Zero();
    sum_state.sum_angular_momentum = Vector3<double>::Zero();
    for (int i = 0; i < num_active_gridpts_; ++i) {
        double mi = get_mass(active_gridpts_[i]);
        const Vector3<double> vi = get_velocity(active_gridpts_[i]);
        const Vector3<double> xi = get_position(active_gridpts_[i]);
        sum_state.sum_mass             += mi;
        sum_state.sum_momentum         += mi*vi;
        sum_state.sum_angular_momentum += mi*xi.cross(vi);
    }
    return sum_state;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
