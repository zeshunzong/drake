#include "drake/multibody/mpm/MPMTransfer.h"

namespace drake {
namespace multibody {
namespace mpm {

void MPMTransfer::SetUpTransfer(SparseGrid* grid,
                                Particles* particles) {

    int num_particles = particles->get_num_particles();
    std::vector<Vector3<int>> batch_indices(num_particles); // batch_indices is the 3Dindex of batch that each particle belongs to
    // Preallocate the indices of batches
    for (int p = 0; p < num_particles; ++p) {
        batch_indices[p] = CalcBatchIndex(particles->get_position(p),
                                          grid->get_h());
        // batch_indices at particle p is the (3-integer) index of the closet grid node
    }
    // If the sparse grid is uninitialized, reserve the space
    if (grid->get_num_active_gridpt() == 0) {
        // TODO(yiminlin.tri): Magic number
        grid->reserve(3*particles->get_num_particles());
    }

    // TODO(yiminlin.tri): expensive... To optimize
    // TODO(simon): batch_indices has the same size as particles, can only use batch_indices but not particles
    // simon: what it does is loop over all particles (equivalently batch for each particle, mark the 27 adjacent grid nodes as active, and sort grid nodes)
    grid->UpdateActiveGridPoints(batch_indices, *particles);
    
    SortParticles(batch_indices, *grid, particles); // sort particles based on sorted grid nodes above

    // TODO(yiminlin.tri): expensive... To optimize
    UpdateBasisAndGradientParticles(*grid, *particles);
    // TODO(yiminlin.tri): Dp_inv_ is hardcoded for quadratic B-Spline
    // The values of Dp_inv_ are different for different B-Spline bases
    // https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf
    Dp_inv_ = 4.0/(grid->get_h()*grid->get_h());
}

void MPMTransfer::TransferParticlesToGrid(const Particles& particles,
                                          SparseGrid* grid) {
    int p_start, p_end, idx_local;
    double mass_p, ref_volume_p;
    int num_active_gridpts = grid->get_num_active_gridpt();
    // Local sum of states m_i v_i f_i on the grid points
    std::vector<std::array<GridState, 27>> local_pads(num_active_gridpts);
    // Positions of grid points in the batch
    std::array<Vector3<double>, 27> batch_positions;
    Vector3<int> batch_index_3d;
    // Clear grid states
    grid->ResetStates();
    // For each batch of particles
    p_start = 0;
    for (int i = 0; i < num_active_gridpts; ++i) {
        if (batch_sizes_[i] != 0) {
            p_end = p_start + batch_sizes_[i];

            // Preallocate positions at grid points in the current batch on a
            // local array
            batch_index_3d = grid->Expand1DIndex(i);
            for (int c = -1; c <= 1; ++c) {
            for (int b = -1; b <= 1; ++b) {
            for (int a = -1; a <= 1; ++a) {
                idx_local = (a+1) + 3*(b+1) + 9*(c+1);
                batch_positions[idx_local] =
                    grid->get_position(batch_index_3d+Vector3<int>(a, b, c));
            }
            }
            }

            // Clear local scratch pad
            for (auto& s : local_pads[i])  {  s.reset(); }

            // For each particle in the batch (Assume particles are sorted with
            // respect to the batch index), accmulate masses, momemtum, and
            // forces into grid points affected by the particle.
            for (int p = p_start; p < p_end; ++p) {
                mass_p = particles.get_mass(p);
                ref_volume_p = particles.get_reference_volume(p);
                // The affine matrix Cp = Bp * Dp^-1
                const Matrix3<double> C_p = particles.get_B_matrix(p)*Dp_inv_;
                AccumulateGridStatesOnBatch(p, mass_p, ref_volume_p,
                                            particles.get_position(p),
                                            particles.get_velocity(p),
                                            C_p,
                                            particles.get_kirchhoff_stress(p),
                                            batch_positions, &local_pads[i]);
                                            // batch_positions, &local_pad);
            }

            p_start = p_end;
        }
    }
    for (int i = 0; i < num_active_gridpts; ++i) {
        if (batch_sizes_[i] != 0) {
            batch_index_3d = grid->Expand1DIndex(i);
            // Put sums of local scratch pads to grid
            WriteBatchStateToGrid(batch_index_3d, local_pads[i], grid);
        }
    }
    // Calculate grid velocities v_i by (mv)_i / m_i
    grid->RescaleVelocities();
}

void MPMTransfer::TransferGridToParticles(const SparseGrid& grid, double dt,
                                          Particles* particles) {
    DRAKE_ASSERT(dt > 0.0);
    int bi, bj, bk, idx_local;
    int p_start, p_end;
    // A local array holding positions and velocities x^{n+1}_i, v^{n+1}_i at a
    // batch
    std::array<BatchState, 27> batch_states;
    Vector3<int> batch_index_3d;
    // For each batch of particles
    p_start = 0;
    for (int i = 0; i < grid.get_num_active_gridpt(); ++i) {
        int batch_size = batch_sizes_[i];
        // Skip empty batches
        if (batch_size == 0) {
            continue;
        }

        p_end = p_start + batch_size;
        // Preallocate positions and velocities at grid points in the current
        // batch on a local array
        batch_index_3d = grid.Expand1DIndex(i);
        bi = batch_index_3d[0];
        bj = batch_index_3d[1];
        bk = batch_index_3d[2];

        for (int c = -1; c <= 1; ++c) {
        for (int b = -1; b <= 1; ++b) {
        for (int a = -1; a <= 1; ++a) {
            idx_local = (a+1) + 3*(b+1) + 9*(c+1);
            Vector3<int> index_3d = Vector3<int>(bi+a, bj+b, bk+c);
            batch_states[idx_local].position = grid.get_position(index_3d);
            batch_states[idx_local].velocity = grid.get_velocity(index_3d);
        }
        }
        }

        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update the particles' states
        for (int p = p_start; p < p_end; ++p) {
            UpdateParticleStates(batch_states, dt, p, particles);
        }

        p_start = p_end;
    }
}

void MPMTransfer::SortParticles(const std::vector<Vector3<int>>& batch_indices,
                                const SparseGrid& grid, Particles* particles) {
    // Vector3<int> batch_idx_3D;
    int num_particles = particles->get_num_particles();
    // A temporary array storing the particle index permutation after sorting
    std::vector<size_t> sorted_indices(num_particles);
    // A temporary array storing the 1D batch index correspond to each particle
    std::vector<int> batch_indices_1D(num_particles);

    for (int p = 0; p < num_particles; ++p) {
        batch_indices_1D[p] = grid.Reduce3DIndex(batch_indices[p]);
    }

    batch_sizes_.resize(grid.get_num_active_gridpt());
    fill(batch_sizes_.begin(), batch_sizes_.end(), 0);
    // Accumulate batch sizes
    for (int p = 0; p < num_particles; ++p) {
        ++batch_sizes_[batch_indices_1D[p]];
    }

    // Calculate the number of batches
    num_batches_ = 0;
    for (int i = 0; i < grid.get_num_active_gridpt(); ++i) {
        if (batch_sizes_[i] > 0) {  num_batches_++;  }
    }

    // start_time = Clock::now();
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
       [&grid, &batch_indices_1D](size_t i1, size_t i2) {
                                return batch_indices_1D[i1] < batch_indices_1D[i2];});

    // Reorder the particles
    particles->Reorder(sorted_indices);
}

void MPMTransfer::UpdateBasisAndGradientParticles(const SparseGrid& grid,
                                                  const Particles& particles) {
    int num_particles, p_start, p_end;
    Vector3<int> batch_index_3d;
    num_particles = particles.get_num_particles();

    bases_val_particles_.reserve(num_particles);
    bases_grad_particles_.reserve(num_particles);

    // For each batch of particles
    p_start = 0;
    // For all active grid points
    for (int i = 0; i < grid.get_num_active_gridpt(); ++i) {
        // Number of particles in the batch centered at the current active grid
        // point.
        int batch_size = batch_sizes_[i];
        // Skip empty batches
        if (batch_size == 0) {
            continue;
        }

        p_end = p_start + batch_size;
        batch_index_3d = grid.Expand1DIndex(i);
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        for (int p = p_start; p < p_end; ++p) {
            const Vector3<double>& xp = particles.get_position(p);
            EvalBasisOnBatch(p, xp, grid, batch_index_3d);
        }
        p_start = p_end;
    }
}

void MPMTransfer::EvalBasisOnBatch(int p, const Vector3<double>& xp,
                                   const SparseGrid& grid,
                                   const Vector3<int>& batch_index_3d) {
    int bi = batch_index_3d[0];
    int bj = batch_index_3d[1];
    int bk = batch_index_3d[2];
    int idx_local;
    Vector3<int> grid_index;

    for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
    for (int a = -1; a <= 1; ++a) {
        grid_index(0) = bi+a;
        grid_index(1) = bj+b;
        grid_index(2) = bk+c;
        idx_local = (a+1) + 3*(b+1) + 9*(c+1);
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        // TODO(yiminlin.tri): Maybe a good idea to preallocate?
        BSpline basis = BSpline(grid.get_h(), grid.get_position(grid_index));
        std::tie(bases_val_particles_[p][idx_local],
                bases_grad_particles_[p][idx_local]) =
        basis.EvalBasisAndGradient(xp);
    }
    }
    }
}

void MPMTransfer::AccumulateGridStatesOnBatch(int p, double m_p,
                                double reference_volume_p,
                                const Vector3<double>& x_p,
                                const Vector3<double>& v_p,
                                const Matrix3<double>& C_p,
                                const Matrix3<double>& tau_p,
                                const std::array<Vector3<double>, 27>&
                                                    batch_positions,
                                std::array<GridState, 27>* local_pad) {
    int idx_local;
    double Ni_p;

    // Accumulate on local scratch pads
    for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
    for (int a = -1; a <= 1; ++a) {
        idx_local = (a+1) + 3*(b+1) + 9*(c+1);
        Ni_p = bases_val_particles_[p][idx_local];
        const Vector3<double>& x_i = batch_positions[idx_local];
        const Vector3<double>& gradNi_p  = bases_grad_particles_[p][idx_local];
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        GridState& state_i = (*local_pad)[idx_local];
        double m_ip = m_p*Ni_p;
        state_i.mass += m_ip;
        // PIC update:
        // state_i.velocity += m_ip*v_p;
        // APIC update:
        state_i.velocity += m_ip*(v_p+C_p*(x_i-x_p));
        state_i.force += -reference_volume_p*tau_p*gradNi_p;
    }
    }
    }
}

void MPMTransfer::WriteBatchStateToGrid(const Vector3<int>& batch_index_3d,
                                const std::array<GridState, 27>& local_pad,
                                SparseGrid* grid) {
    int bi = batch_index_3d[0];
    int bj = batch_index_3d[1];
    int bk = batch_index_3d[2];
    int idx_local;
    Vector3<int> grid_index;

    // Put local scratch pad states into the grid
    for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
    for (int a = -1; a <= 1; ++a) {
        grid_index(0) = bi+a;
        grid_index(1) = bj+b;
        grid_index(2) = bk+c;
        idx_local = (a+1) + 3*(b+1) + 9*(c+1);
        const GridState& state_i = local_pad[idx_local];
        grid->AccumulateMass(grid_index, state_i.mass);
        grid->AccumulateVelocity(grid_index, state_i.velocity);
        grid->AccumulateForce(grid_index, state_i.force);
    }
    }
    }
}

void MPMTransfer::UpdateParticleStates(const std::array<BatchState, 27>&
                                       batch_states,
                                       double dt, int p,
                                       Particles* particles) {
    int idx_local;
    double Ni_p;
    // vp_new_i = v_i^{n+1} * N_i(x_p)
    Vector3<double> vp_new_i;
    Vector3<double> xp = particles->get_position(p);

    // Scratch vectors and matrices
    Vector3<double> vp_new = Vector3<double>::Zero();
    Matrix3<double> Bp_new = Matrix3<double>::Zero();
    Matrix3<double> grad_vp_new = Matrix3<double>::Zero();

    // For each grid node affecting the current particle
    for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
    for (int a = -1; a <= 1; ++a) {
        idx_local = (a+1) + 3*(b+1) + 9*(c+1);
        Ni_p = bases_val_particles_[p][idx_local];
        const Vector3<double>& gradNi_p = bases_grad_particles_[p][idx_local];
        const Vector3<double>& xi = batch_states[idx_local].position;
        const Vector3<double>& vi_new = batch_states[idx_local].velocity;
        vp_new_i = vi_new*Ni_p;
        // v_p^{n+1} = \sum v_i^{n+1} N_i(x_p)
        vp_new += vp_new_i;
        // B_p^{n+1} = \sum v_i^{n+1}*(x_i - x_p^n)^T N_i(x_p)
        Bp_new += vp_new_i*(xi-xp).transpose();
        // Accumulate grad_vp_new: F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
        grad_vp_new += vi_new*gradNi_p.transpose();
    }
    }
    }

    // Update velocities and elastic deformation gradients of the particle
    // F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
    // Note that we assume the plastic deformation gradient doesn't change
    // during the evolution.
    particles->set_elastic_deformation_gradient(p,
                        (Matrix3<double>::Identity() + dt*grad_vp_new)
                        *particles->get_elastic_deformation_gradient(p));
    particles->set_velocity(p, vp_new);
    particles->set_B_matrix(p, Bp_new);
}

Vector3<int> MPMTransfer::CalcBatchIndex(const Vector3<double>& xp, double h)
                                                                        const {
    return Vector3<int>(std::round(xp(0)/h), std::round(xp(1)/h),
                        std::round(xp(2)/h));
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
