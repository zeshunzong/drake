#include "drake/multibody/mpm/SparseGrid.h"
#include <iostream>
namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
SparseGrid<T>::SparseGrid(T h): num_active_gridpts_(0), h_(h),
                                  active_gridpts_(0) {
    std::cout << "create grid with h = " << h << std::endl;
                                  }

template <typename T>
void SparseGrid<T>::reserve(size_t capacity) {
    index_map_.reserve(capacity);
    active_gridpts_.reserve(capacity);
    velocities_.reserve(capacity);
    masses_.reserve(capacity);
    forces_.reserve(capacity);
}

template <typename T>
int SparseGrid<T>::get_num_active_gridpt() const {
    return num_active_gridpts_;
}

template <typename T>
T SparseGrid<T>::get_h() const {
    return h_;
}

template <typename T>
Vector3<T> SparseGrid<T>::get_position(const Vector3<int>& index_3d) const {
    return Vector3<T>(index_3d(0)*h_, index_3d(1)*h_, index_3d(2)*h_);
}

template <typename T>
const Vector3<T>& SparseGrid<T>::get_velocity(const Vector3<int>& index_3d)
                                                                        const {
    return velocities_[Reduce3DIndex(index_3d)];
}

template <typename T>
T SparseGrid<T>::get_mass(const Vector3<int>& index_3d) const {
    return masses_[Reduce3DIndex(index_3d)];
}

template <typename T>
const Vector3<T>& SparseGrid<T>::get_force(const Vector3<int>& index_3d)
                                                                        const {
    return forces_[Reduce3DIndex(index_3d)];
}

template <typename T>
const Vector3<T>& SparseGrid<T>::get_force(size_t index_1d)
                                                                        const {
    return forces_[index_1d];
}

template <typename T>
const Vector3<T>& SparseGrid<T>::get_velocity(size_t index_1d) const {
    return velocities_[index_1d];
}

template <typename T>
void SparseGrid<T>::set_velocity(const Vector3<int>& index_3d,
                        const Vector3<T>& velocity) {
    velocities_[Reduce3DIndex(index_3d)] = velocity;
}

template <typename T>
void SparseGrid<T>::set_mass(const Vector3<int>& index_3d, T mass) {
    masses_[Reduce3DIndex(index_3d)] = mass;
}

template <typename T>
void SparseGrid<T>::set_force(const Vector3<int>& index_3d,
                     const Vector3<T>& force) {
    forces_[Reduce3DIndex(index_3d)] = force;
}

template <typename T>
void SparseGrid<T>::set_velocity(size_t index_1d,
                              const Vector3<T>& velocity) {
    velocities_[index_1d] = velocity;
}

template <typename T>
void SparseGrid<T>::AccumulateVelocity(const Vector3<int>& index_3d,
                                    const Vector3<T>& velocity) {
    velocities_[Reduce3DIndex(index_3d)] += velocity;
}

template <typename T>
void SparseGrid<T>::AccumulateMass(const Vector3<int>& index_3d, T mass) {
    masses_[Reduce3DIndex(index_3d)] += mass;
}

template <typename T>
void SparseGrid<T>::AccumulateForce(const Vector3<int>& index_3d,
                                 const Vector3<T>& force) {
    forces_[Reduce3DIndex(index_3d)] += force;
}

template <typename T>
void SparseGrid<T>::ApplyGravitationalForces(T dt, Vector3<T>& gravitational_acceleration) {
      Vector3<T> dv = dt*gravitational_acceleration;
      for (int i = 0; i < get_num_active_gridpt(); ++i) {
        const Vector3<T>& velocity_i = get_velocity(i);
        set_velocity(i, velocity_i + dv);
    }
}

template <typename T>
bool SparseGrid<T>::is_active(const Vector3<int>& gridpt) const {
    return index_map_.count(gridpt) == 1;
}

template <typename T>
void SparseGrid<T>::UpdateActiveGridPoints(
                            const std::vector<Vector3<int>>& active_gridpts) {
    num_active_gridpts_ = active_gridpts.size();
    for (int i = 0; i < num_active_gridpts_; ++i) {
        active_gridpts_[i] = active_gridpts[i];
        index_map_[active_gridpts_[i]] = i;
    }
}

template <typename T>
void SparseGrid<T>::UpdateActiveGridPoints(const std::vector<Vector3<int>>& batch_indices,
                                        const Particles<T>& particles) {
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
                active_gridpts_.push_back(gridpt);
                num_active_gridpts_++;
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
    // resize
    masses_.resize(num_active_gridpts_);
    velocities_.resize(num_active_gridpts_);
    forces_.resize(num_active_gridpts_);
}

template <typename T>
void SparseGrid<T>::RescaleVelocities() {
    for (int i = 0; i < num_active_gridpts_; ++i) {
        velocities_[i] = velocities_[i] / masses_[i];
    }
}

template <typename T>
void SparseGrid<T>::ResetStates() {
    DRAKE_ASSERT(masses_.size() == num_active_gridpts_);
    DRAKE_ASSERT(velocities_.size() == num_active_gridpts_);
    DRAKE_ASSERT(forces_.size() == num_active_gridpts_);
    std::fill(masses_.begin(), masses_.begin()+num_active_gridpts_, 0.0);
    std::fill(velocities_.begin(), velocities_.begin()+num_active_gridpts_,
              Vector3<T>::Zero());
    std::fill(forces_.begin(), forces_.begin()+num_active_gridpts_,
              Vector3<T>::Zero());
}

template <typename T>
size_t SparseGrid<T>::Reduce3DIndex(const Vector3<int>& index_3d) const {
    return index_map_.at(index_3d);
}

template <typename T>
Vector3<int> SparseGrid<T>::Expand1DIndex(size_t index_1D) const {
    return active_gridpts_[index_1D];
}

template <typename T>
void SparseGrid<T>::UpdateVelocity(T dt) {
    for (int i = 0; i < num_active_gridpts_; ++i) {
        velocities_[i] += dt*forces_[i]/masses_[i];
    }
}

template <typename T>
void SparseGrid<T>::OverwriteGridVelocity(std::function<void(Vector3<T>,
                                                          T,
                                                          Vector3<T>*)>
                                       update_velocity, T t) {
    // For all grid points, overwrite grid velocity
    for (int i = 0; i < num_active_gridpts_; ++i) {
        update_velocity(get_position(active_gridpts_[i]), t, &velocities_[i]);
    }
}

template <typename T>
TotalMassEnergyMomentum<T> SparseGrid<T>::GetTotalMassAndMomentum() const {
    TotalMassEnergyMomentum<T> sum_state;
    sum_state.sum_mass             = 0.0;
    sum_state.sum_momentum         = Vector3<T>::Zero();
    sum_state.sum_angular_momentum = Vector3<T>::Zero();
    for (int i = 0; i < num_active_gridpts_; ++i) {
        T mi = get_mass(active_gridpts_[i]);
        const Vector3<T> vi = get_velocity(active_gridpts_[i]);
        const Vector3<T> xi = get_position(active_gridpts_[i]);
        sum_state.sum_mass             += mi;
        sum_state.sum_momentum         += mi*vi;
        sum_state.sum_angular_momentum += mi*xi.cross(vi);
    }
    return sum_state;
}

template class SparseGrid<double>;
template class SparseGrid<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
