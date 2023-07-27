#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/Particles.h"
#include "drake/multibody/mpm/SparseGrid.h"

namespace drake {
namespace multibody {
namespace mpm {

// A implementation of MPM's Particles to Grid (P2G) and Grid to Particles (G2P)
// operations
template <typename T>
class MPMTransfer {
 public:
    MPMTransfer() {}

    // Sort the particles according to the batch number, and preallocate
    // bases' evaluations for transfer routines. This routine has to be called
    // at the beginning of each time step (before P2G and G2P transfers),
    // otherwise the results will be incorrect.
    void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles);

    // Transfer masses, velocities, and Kirchhoff stresses on the particles
    // to masses, velocities, and forces on the grid
    void TransferParticlesToGrid(const Particles<T>& particles, SparseGrid<T>* grid);

    // Transfer velocities on the grids to velocities and deformation
    // gradients on the particles
    void TransferGridToParticles(const SparseGrid<T>& grid, T dt,
                                 Particles<T>* particles);

 private:
    friend class MPMTransferTest;

    // A struct holding the accmulated grid state on a batch when transferring
    // from particles to grid.
    struct GridState {
        T mass;
        Vector3<T> velocity;
        Vector3<T> force;

        void reset() {
            mass = 0.0;
            velocity.setZero();
            force.setZero();
        }
    };

    // A struct holding information in a batch when transferring from grid to
    // particles. We use the position of a grid point to update the B matrix of
    // a particle, and we use the velocity of a grid point to update the
    // velocities and gradients of velocities of particles.
    struct BatchState {
        Vector3<T> position;
        Vector3<T> velocity;
    };

    // Sort the particles according to the batch number, in increasing order.
    // As below shown, o denotes the grid points, $ denotes the batch centered
    // around the grid point. # of batch = # of grid points
    // o = = = o = = = o = = = o = = = o
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    // o = = = o = = = o = = = o = = = o
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    // o = = = o = = = o = = = o = = = o
    //||      ||      ||      ||      ||
    //||      ||   $$$$$$$$   ||      ||
    //||      ||   $  ||  $   ||      ||
    // o = = = o = $ = o =$= = o = = = o
    //||      ||   $  ||  $   ||      ||
    //||      ||   $$$$$$$$   ||      ||
    //||      ||      ||      ||      ||
    // o = = = o = = = o = = = o = = = o
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    // o = = = o = = = o = = = o = = = o
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    //||      ||      ||      ||      ||
    // o = = = o = = = o = = = o = = = o
    // The batches are ordered in a lexiographical ordering, similar to grid
    // points.
    // SortParticles also initialize the sparse grid with active grid points
    void SortParticles(const std::vector<Vector3<int>>& batch_indices,
                       const SparseGrid<T>& grid, Particles<T>* particles);

    // Update the evalutions and gradients of BSpline bases on each particle,
    // and update bases_val_particles_ and bases_grad_particles_
    void UpdateBasisAndGradientParticles(const SparseGrid<T>& grid,
                                         const Particles<T>& particles);

    // Evaluate (27) bases neighboring to the given batch, at the p-th particle
    // with position xp, and put the results into preallocated vectors
    void EvalBasisOnBatch(int p, const Vector3<T>& xp,
                          const SparseGrid<T>& grid,
                          const Vector3<int>& batch_index_3d);

    // At a particular particle p in batch with batch_index_3d, transfer
    // particle states (m, mv, tau) to (m, mv, f). Note that we temporarily
    // store the momentum into particles' velocities, in TransferParticlesToGrid
    // we will scale the momentum with the updated mass to get the velocities.
    // We update the velocity according to affine particle-in-cell methods,
    // where we can approximate the velocity field at a grid point with an
    // affine approximation around the particle:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    // B_p is a 3x3 matrix stored in Particles and D_p^-1 is a constant depends
    // on the grid size. D_p^-1 is stored in the class as a member variable.
    // We refer C_p as the affine matrix.
    void AccumulateGridStatesOnBatch(int p, T mass_p,
                                     T reference_volume_p,
                                     const Vector3<T>& position_p,
                                     const Vector3<T>& velocity_p,
                                     const Matrix3<T>& affine_matrix_p,
                                     const Matrix3<T>& tau_p,
                                     const std::array<Vector3<T>, 27>&
                                                            batch_positions,
                                     std::array<GridState, 27>* sum_local);

    void WriteBatchStateToGrid(const Vector3<int>& batch_index_3d,
                               const std::array<GridState, 27>& sum_local,
                               SparseGrid<T>* grid);

    // Update particle states F_p^{n+1} and v_p^{n+1}
    void UpdateParticleStates(const std::array<BatchState, 27>& batch_states,
                              T dt, int p,
                              Particles<T>* particles);

    // Given the position of a particle xp, calculate the index of the batch
    // this particle is in.
    Vector3<int> CalcBatchIndex(const Vector3<T>& xp, T h) const;

    // The inverse value of diagonal entries of the matrix D_p in APIC
    T Dp_inv_;

    // Evaluations and gradients of BSpline bases on each particle
    // i.e. N_i(x_p), \nabla N_i(x_p)
    // Length of the vector = # of particles.
    // Length of an element in the vector = 27 (max # of affected grid nodes)
    std::vector<std::array<T, 27>> bases_val_particles_{};
    std::vector<std::array<Vector3<T>, 27>> bases_grad_particles_{};
    // A vector holding the number of particles inside each batch
    std::vector<int> batch_sizes_{};
    int num_batches_;
};  // class MPMTransfer

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
