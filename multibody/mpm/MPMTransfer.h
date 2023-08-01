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
#include <iomanip> // For setprecision
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


    // reserve sparse grid, find active grid points, sort particles
    int MakeGridCompatibleWithParticles(Particles<T>* particles, SparseGrid<T>* grid) {
        int num_particles = particles->get_num_particles(); 
        std::vector<Vector3<int>> batch_indices(
            num_particles);  // batch_indices is the 3Dindex of batch that each particle belongs to
        // Preallocate the indices of batches
        for (int p = 0; p < num_particles; ++p) {
            batch_indices[p] =
                CalcBatchIndex(particles->get_position(p), grid->get_h());
            // batch_indices at particle p is the (3-integer) index of the closet grid node
        }
        // If the sparse grid is uninitialized, reserve the space
        if (grid->get_num_active_gridpt() == 0) {
            grid->reserve(3 * particles->get_num_particles()); // TODO(yiminlin.tri): Magic number
        }
        // TODO(yiminlin.tri): expensive... To optimize
        // TODO(simon): batch indices has the same size as particles, can only use
        // batch_indices but not particles simon:
        grid->UpdateActiveGridPoints(batch_indices, *particles);

        SortParticles(batch_indices, *grid,
                        particles);  // sort particles based on sorted grid nodes above
        UpdateBasisAndGradientParticles(*grid, *particles);

        return grid->get_num_active_gridpt(); 
    }

    // write grid velocities


    T computeEnergyForceHessian(Particles<T>* particles, SparseGrid<T>* grid, T dt){
        ComputeWeightsAndWeightsGradients(*particles, *grid);
        ComputeParticleFnewG2P(*grid, dt, particles); //Fnew(vi*) write to particles
        T energy = ComputeParticleEnergy(*particles);
        ComputeForceP2G(*particles, grid); // write grid force and hessian onto grid
        return energy;
    } 

    void ComputeWeightsAndWeightsGradients(const Particles<T>& particles, const SparseGrid<T>& grid) {
        UpdateBasisAndGradientParticles(grid, particles);
        // TODO(yiminlin.tri): Dp_inv_ is hardcoded for quadratic B-Spline
        Dp_inv_ = 4.0 / (grid.get_h() * grid.get_h());
    }

    // g2p way to compute particle Fnew(vi*, Fp0, dt)
    // Fp_new = Fₚ(vᵢ*) = [I + ∑ᵢ (xᵢ*−xᵢ⁰) ∇ Nᵢ(xₚ⁰)ᵀ]F⁰ₚ
    void ComputeParticleFnewG2P(const SparseGrid<T>& grid, T dt, Particles<T>* particles) {
        particles->resize_elastic_deformation_gradient_new(particles->get_num_particles());
        int bi, bj, bk, idx_local; int p_start, p_end;
        // A local array holding positions and velocities x^{n+1}_i, v^{n+1}_i at a batch
        std::array<BatchState, 27> batch_states;
        Vector3<int> batch_index_3d;
        // For each batch of particles
        p_start = 0;
        for (int i = 0; i < grid.get_num_active_gridpt(); ++i) {
            int batch_size = batch_sizes_[i];
            if (batch_size == 0) {continue;} // Skip empty batches
            p_end = p_start + batch_size;
            // Preallocate positions and velocities at grid points in the current
            // batch on a local array
            batch_index_3d = grid.Expand1DIndex(i);
            bi = batch_index_3d[0]; bj = batch_index_3d[1]; bk = batch_index_3d[2];
            for (int c = -1; c <= 1; ++c) {
            for (int b = -1; b <= 1; ++b) {
            for (int a = -1; a <= 1; ++a) {
                idx_local = (a+1) + 3*(b+1) + 9*(c+1);
                Vector3<int> index_3d = Vector3<int>(bi+a, bj+b, bk+c);
                batch_states[idx_local].position = grid.get_position(index_3d);
                batch_states[idx_local].velocity = grid.get_velocity(index_3d);
            }}}
            // For each particle in the batch (Assume particles are sorted with
            // respect to the batch index), update the particles' Fnew(v*)
            for (int p = p_start; p < p_end; ++p) {
                UpdateParticlesFnew(batch_states, dt, p, particles);
            }
            p_start = p_end;
        }
    }
    // compute energy = sum_p V0 psi(F(v*)), the energy will be a function of v*
    T ComputeParticleEnergy(const Particles<T>& particles){
        // static_cast<int>(size);
        T energy = 0;
        for (int p = 0; p < particles.get_num_particles(); ++p) {
            T volp = particles.get_reference_volume(p);
            energy = energy + volp* particles.CalcParticleEnergyDensity(p, particles.get_elastic_deformation_gradient_new(p));
        }
        return energy;
    }

    // Hessian TBD
    void ComputeForceP2G(const Particles<T>& particles, SparseGrid<T>* grid) {
        int p_start, p_end, idx_local;
        int num_active_gridpts = grid->get_num_active_gridpt();
        // Local sum of states m_i v_i f_i on the grid points
        std::vector<std::array<GridState, 27>> local_pads(num_active_gridpts);
        // Positions of grid points in the batch
        std::array<Vector3<T>, 27> batch_positions;
        Vector3<int> batch_index_3d;
        // Clear grid states
        grid->ResetStates();
        // For each batch of particles
        p_start = 0;
        for (int i = 0; i < num_active_gridpts; ++i) {
            if (batch_sizes_[i] != 0) {
                p_end = p_start + batch_sizes_[i];
            // Preallocate positions at grid points in the current batch on a local array
                batch_index_3d = grid->Expand1DIndex(i);
                for (int c = -1; c <= 1; ++c) {
                for (int b = -1; b <= 1; ++b) {
                for (int a = -1; a <= 1; ++a) {
                    idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
                    batch_positions[idx_local] = grid->get_position(batch_index_3d + Vector3<int>(a, b, c));
                }}}
                for (auto& s : local_pads[i]) {s.reset();}  // Clear local scratch pad
                // For each particle in the batch, accumulate force
                for (int p = p_start; p < p_end; ++p) {
                    // set first PK stress as computed using F(v*)
                    Matrix3<T> PK_stress_new;
                    particles.compute_first_PK_stress(p, particles.get_elastic_deformation_gradient_new(p), &PK_stress_new);
                    // update force via first PK stress
                    AccumulateGridForcesOnBatch(p, particles.get_reference_volume(p), PK_stress_new,
                        particles.get_elastic_deformation_gradient(p), &local_pads[i]);
                }
                p_start = p_end;
            }
        }
        for (int i = 0; i < num_active_gridpts; ++i) {
            if (batch_sizes_[i] != 0) {
            batch_index_3d = grid->Expand1DIndex(i);
            WriteBatchForceToGrid(batch_index_3d, local_pads[i], grid);
            }
        }
    }

    void ComputeHessianP2G(Particles<T>& particles, SparseGrid<T>* grid, MatrixX<T>* hessian) {
        
        hessian->resize(grid->get_num_active_gridpt()*3, grid->get_num_active_gridpt()*3);
        hessian->setZero();
        particles.resize_stress_derivatives(particles.get_num_particles());
        particles.resize_stress_derivatives_contractF_contractF(particles.get_num_particles());
        particles.ComputePiolaDerivatives();
        particles.ContractPiolaDerivativesWithFWithF();
        int p_start, p_end;// idx_local;
        int num_active_gridpts = grid->get_num_active_gridpt();
        Vector3<int> batch_index_3d;
        
        // For each batch i*
        p_start = 0;
        for (int i_star = 0; i_star < num_active_gridpts; ++i_star) {
            if (batch_sizes_[i_star] != 0) {
                
                MatrixX<T> pad_hessian; pad_hessian.resize(27*3, 27*3); pad_hessian.setZero();
                p_end = p_start + batch_sizes_[i_star];
                // Preallocate positions at grid points in the current batch on a local array
                batch_index_3d = grid->Expand1DIndex(i_star);
                // For each particle in the batch, 
                for (int p = p_start; p < p_end; ++p) {
                    // loop over each i of 27 nodes in the numerator
                    for (size_t i = 0; i < 27; ++i){
                        // loop over three dimensions α
                        for (size_t alpha = 0; alpha < 3; ++alpha){
                            // loop over each j of 27 nodes in the denominator
                            for (size_t j = 0; j < 27; ++j){
                                // loop over three dimensions τ
                                for (size_t tau = 0; tau < 3; ++tau){

                                    const Vector3<T>& gradNi_p = bases_grad_particles_[p][i];
                                    const Vector3<T>& gradNj_p = bases_grad_particles_[p][j];
                                    Eigen::Matrix<T, 9, 9>& A_alphabeta_taulambda = particles.get_stress_derivatives_contractF_contractF_(p);
                                    // for the 9by9 matrix A_alphabeta_taulambda, we set the convention to be A(alpha*3+beta, tau*3+lambda)
                                    for (size_t beta = 0; beta<3; ++beta) {
                                        for (size_t lambda = 0; lambda<3; ++lambda) {
                                            pad_hessian(3*i+alpha, 3*j+tau) += gradNi_p(beta) * A_alphabeta_taulambda(beta*3+alpha, lambda*3+tau) * gradNj_p(lambda);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                p_start = p_end;

                // print_matrix(pad_hessian);
            
                // assign local hessian to global, local hessian is (27*3) by (27*3)
                for (int ic = -1; ic <= 1; ++ic) {
                for (int ib = -1; ib <= 1; ++ib) {
                for (int ia = -1; ia <= 1; ++ia) {
                    int idx_local_i = (ia + 1) + 3 * (ib + 1) + 9 * (ic + 1); // local index 0-26
                    Vector3<int> node_i_index_3d = batch_index_3d + Vector3<int>(ia, ib, ic); // global 3d index of node i

                    for (int jc = -1; jc <= 1; ++jc) {
                    for (int jb = -1; jb <= 1; ++jb) {
                    for (int ja = -1; ja <= 1; ++ja) {
                        int idx_local_j = (ja + 1) + 3 * (jb + 1) + 9 * (jc + 1); // local index 0-26
                        Vector3<int> node_j_index_3d = batch_index_3d + Vector3<int>(ja, jb, jc); // global 3d index of node i

                        MatrixX<T> dfi_dvj_block = pad_hessian.block(idx_local_i*3, idx_local_j*3, 3, 3);

                        (*hessian).block(grid->Reduce3DIndex(node_i_index_3d)*3, grid->Reduce3DIndex(node_j_index_3d)*3, 3, 3) += dfi_dvj_block;
                    }}}
                }}}
                
            }
        }
    }


    int DerivativeTest1(SparseGrid<T>* grid, Particles<T>* particles) {
        SetUpTransfer(grid, particles); // wip should also be computed here
        return grid->get_num_active_gridpt();                        
    }

    void DerivativeTest2(SparseGrid<T>* grid, const std::vector<Vector3<T>>& grid_velocities_input) {
        // DRAKE_DEMAND(grid_velocities_input.size() == grid->get_num_active_gridpt());
        for (size_t i = 0; i < grid_velocities_input.size(); i++) {
            grid->set_velocity(i, grid_velocities_input[i]);
        }
        // autodiff indep variables grid velocities passed to grid.
    }
    void DerivativeTest3(const SparseGrid<T>& grid, T dt, Particles<T>* particles){
        // copied from regular g2p, only thing that is changed is that (only F_tmp_p is updated)
        particles->resize_elastic_deformation_gradient_tmp(particles->get_num_particles()); //F_temp is not allocated in constructor right now
        int bi, bj, bk, idx_local;
        int p_start, p_end;
        // A local array holding positions and velocities x^{n+1}_i, v^{n+1}_i at a batch
        std::array<BatchState, 27> batch_states;
        Vector3<int> batch_index_3d;
        // For each batch of particles
        p_start = 0;
        for (int i = 0; i < grid.get_num_active_gridpt(); ++i) {
            int batch_size = batch_sizes_[i];
            if (batch_size == 0) {continue;} // Skip empty batches
            p_end = p_start + batch_size;
            // Preallocate positions and velocities at grid points in the current
            // batch on a local array
            batch_index_3d = grid.Expand1DIndex(i);
            bi = batch_index_3d[0]; bj = batch_index_3d[1]; bk = batch_index_3d[2];
            for (int c = -1; c <= 1; ++c) {
            for (int b = -1; b <= 1; ++b) {
            for (int a = -1; a <= 1; ++a) {
                idx_local = (a+1) + 3*(b+1) + 9*(c+1);
                Vector3<int> index_3d = Vector3<int>(bi+a, bj+b, bk+c);
                batch_states[idx_local].position = grid.get_position(index_3d);
                batch_states[idx_local].velocity = grid.get_velocity(index_3d);
            }}}
            // For each particle in the batch (Assume particles are sorted with
            // respect to the batch index), update the particles' Ftmp
            for (int p = p_start; p < p_end; ++p) {
                UpdateParticlesFtmp(batch_states, dt, p, particles);
            }
            p_start = p_end;
        }
    }

    void print_matrix(MatrixX<T>& M) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "begin printing matrix ---------------\n";
        for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                std::cout << M(i,j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "end printing matrix ---------------\n";
    }
    // compute energy = sum_p V0 psi(F(v*)), the energy will be a function of v*
    T DerivativeTest4(const Particles<T>& particles) {
        T energy = 0;
        for (int p = 0; p < particles.get_num_particles(); ++p) {
            T volp = particles.get_reference_volume(p);
            energy = energy + volp* particles.CalcParticleEnergyDensity(p, particles.get_elastic_deformation_gradient_tmp(p));
        }
        return energy;
    }

    void DerivativeTest5(const Particles<T>& particles, SparseGrid<T>* grid){
        int p_start, p_end, idx_local;
        int num_active_gridpts = grid->get_num_active_gridpt();
        // Local sum of states m_i v_i f_i on the grid points
        std::vector<std::array<GridState, 27>> local_pads(num_active_gridpts);
        // Positions of grid points in the batch
        std::array<Vector3<T>, 27> batch_positions;
        Vector3<int> batch_index_3d;
        // Clear grid states
        grid->ResetStates();
        // For each batch of particles
        p_start = 0;
        for (int i = 0; i < num_active_gridpts; ++i) {
            if (batch_sizes_[i] != 0) {
                p_end = p_start + batch_sizes_[i];

            // Preallocate positions at grid points in the current batch on a local array
                batch_index_3d = grid->Expand1DIndex(i);
                for (int c = -1; c <= 1; ++c) {
                for (int b = -1; b <= 1; ++b) {
                for (int a = -1; a <= 1; ++a) {
                    idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
                    batch_positions[idx_local] = grid->get_position(batch_index_3d + Vector3<int>(a, b, c));
                }}}

                for (auto& s : local_pads[i]) {s.reset();}  // Clear local scratch pad

                // For each particle in the batch, accumulate force
                for (int p = p_start; p < p_end; ++p) {
                    // set first PK stress as computed using F(v*)
                    Matrix3<T> PK_stress_new;
                    particles.compute_first_PK_stress(p, particles.get_elastic_deformation_gradient_tmp(p), &PK_stress_new);
                    // update force via first PK stress
                    AccumulateGridForcesOnBatchFirstPKTest(p, particles.get_reference_volume(p),
                        particles.get_elastic_deformation_gradient(p),
                        PK_stress_new, &local_pads[i]);
                }
                p_start = p_end;
            }
        }
        for (int i = 0; i < num_active_gridpts; ++i) {
            if (batch_sizes_[i] != 0) {
            batch_index_3d = grid->Expand1DIndex(i);
            WriteBatchForceToGrid(batch_index_3d, local_pads[i], grid);
            }
        }
    }

    
    // 0. Setup a few particles, with xp, Fp. vp is irrelavent
    // 1. Setup Transfer to get number of active grids and wip
    // 2. overwrite grid velocities v* to be indep variables (just like grid update)
    // 3. g2p way to compute Fp(v*), using the wip computed in step 2. do not update everything else in particles
    // 4. compute energy by summing over particles
    // 5. p2g way to compute grid force (using orifinal Fp as derivative, using Fp(v*) for first_PK_stress)


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
    void AccumulateGridStatesOnBatchKirchhoff(int p, T mass_p,
                                     T reference_volume_p,
                                     const Vector3<T>& position_p,
                                     const Vector3<T>& velocity_p,
                                     const Matrix3<T>& affine_matrix_p,
                                     const Matrix3<T>& tau_p,
                                     const std::array<Vector3<T>, 27>&
                                                            batch_positions,
                                     std::array<GridState, 27>* local_pad);

    // Note: here P_p might not be computed from FE_p. FE_p is just for chain rule in derivative
    void AccumulateGridStatesOnBatchFirstPK(int p, T mass_p,
                                     T reference_volume_p,
                                     const Vector3<T>& position_p,
                                     const Vector3<T>& velocity_p,
                                     const Matrix3<T>& affine_matrix_p,
                                     const Matrix3<T>& FE_p,
                                     const Matrix3<T>& P_p,
                                     const std::array<Vector3<T>, 27>&
                                                            batch_positions,
                                     std::array<GridState, 27>* local_pad);

    void AccumulateGridForcesOnBatchFirstPKTest(int p, T reference_volume_p, const Matrix3<T>& FE_p, const Matrix3<T>& P_p,
                                      std::array<GridState, 27>* local_pad){
         // mostly copy from p2g, but we only compute grid force, using Fp(v*) for first PK stress                               
            int idx_local;
            // Accumulate on local scratch pads
            for (int c = -1; c <= 1; ++c) {
                for (int b = -1; b <= 1; ++b) {
                    for (int a = -1; a <= 1; ++a) {
                        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
                        const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
                        // For each particle in the batch
                        GridState& state_i = (*local_pad)[idx_local];
                        state_i.force += -reference_volume_p * P_p * FE_p.transpose() * gradNi_p;
            }}}                   
    }

    //fᵢ += -V^0_p \cdot PKstress \cdot  F_p^T \cdot \nabla N_i(x_p)
    void AccumulateGridForcesOnBatch(int p, T reference_volume_p, const Matrix3<T>& PK_stress, const Matrix3<T>& FE,
                                     std::array<GridState, 27>* local_pad){
        int idx_local;
        // Accumulate on local scratch pads
        for (int c = -1; c <= 1; ++c) {
            for (int b = -1; b <= 1; ++b) {
                for (int a = -1; a <= 1; ++a) {
                    idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
                    const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
                    // For each particle in the batch
                    GridState& state_i = (*local_pad)[idx_local];
                    state_i.force += -reference_volume_p * PK_stress * FE.transpose() * gradNi_p;
        }}}                   
    }

    void WriteBatchStateToGrid(const Vector3<int>& batch_index_3d,
                               const std::array<GridState, 27>& sum_local,
                               SparseGrid<T>* grid);

    void WriteBatchForceToGrid(const Vector3<int>& batch_index_3d,
                               const std::array<GridState, 27>& sum_local,
                               SparseGrid<T>* grid);

    // Update particle states F_p^{n+1} and v_p^{n+1}
    void UpdateParticleStates(const std::array<BatchState, 27>& batch_states,
                              T dt, int p,
                              Particles<T>* particles);

    // Given the position of a particle xp, calculate the index of the batch
    // this particle is in.
    Vector3<int> CalcBatchIndex(const Vector3<T>& xp, T h) const;


    // like a regular F update, but store in Fnew, since we want to keep the original F
    void UpdateParticlesFnew(const std::array<BatchState, 27>& batch_states, T dt, int p, Particles<T>* particles) {
        int idx_local;
        Matrix3<T> grad_vp_new = Matrix3<T>::Zero();
        // For each grid node affecting the current particle
        for (int c = -1; c <= 1; ++c) {
        for (int b = -1; b <= 1; ++b) {
        for (int a = -1; a <= 1; ++a) {
            idx_local = (a+1) + 3*(b+1) + 9*(c+1);
            const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
            const Vector3<T>& vi_new = batch_states[idx_local].velocity;
            // Accumulate grad_vp_new
            grad_vp_new += vi_new*gradNi_p.transpose();
        }}}
        // F_p^{v*} = (I + dt*grad_vp_new)*F_p^n
        particles->set_elastic_deformation_gradient_new(p,
                            (Matrix3<T>::Identity() + dt*grad_vp_new)*particles->get_elastic_deformation_gradient(p));
    }


    // // like a regular F update, but store in Ftmp, since we want to keep the original F
    void UpdateParticlesFtmp(const std::array<BatchState, 27>& batch_states, T dt, int p, Particles<T>* particles) {
        int idx_local;
        Matrix3<T> grad_vp_new = Matrix3<T>::Zero();
        // For each grid node affecting the current particle
        for (int c = -1; c <= 1; ++c) {
        for (int b = -1; b <= 1; ++b) {
        for (int a = -1; a <= 1; ++a) {
            idx_local = (a+1) + 3*(b+1) + 9*(c+1);
            const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
            const Vector3<T>& vi_new = batch_states[idx_local].velocity;
            // Accumulate grad_vp_new: F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
            grad_vp_new += vi_new*gradNi_p.transpose();
        }}}
        // F_p^{v*} = (I + dt*grad_vp_new)*F_p^n
        particles->set_elastic_deformation_gradient_tmp(p,
                            (Matrix3<T>::Identity() + dt*grad_vp_new)*particles->get_elastic_deformation_gradient(p));
    }

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
