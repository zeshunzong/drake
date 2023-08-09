#include "drake/multibody/mpm/MPMTransfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MPMTransfer<T>::SetUpTransfer(SparseGrid<T>* grid,
                                   Particles<T>* particles) {
  int num_particles = particles->get_num_particles(); 
  std::vector<Vector3<int>> batch_indices(
      num_particles);  // batch_indices is the 3Dindex of batch that each
                       // particle belongs to
  // Preallocate the indices of batches
  for (int p = 0; p < num_particles; ++p) {
    batch_indices[p] =
        CalcBatchIndex(particles->get_position(p), grid->get_h());
    // batch_indices at particle p is the (3-integer) index of the closet grid
    // node
  }
  // If the sparse grid is uninitialized, reserve the space
  if (grid->get_num_active_gridpt() == 0) {
    // TODO(yiminlin.tri): Magic number
    grid->reserve(3 * particles->get_num_particles());
  }
  // TODO(yiminlin.tri): expensive... To optimize
  // TODO(simon): batch_indices has the same size as particles, can only use
  // batch_indices but not particles simon: what it does is loop over all
  // particles (equivalently batch for each particle, mark the 27 adjacent grid
  // nodes as active, and sort grid nodes)
  grid->UpdateActiveGridPoints(batch_indices, *particles);

  SortParticles(batch_indices, *grid,
                particles);  // sort particles based on sorted grid nodes above
  // TODO(yiminlin.tri): expensive... To optimize
  UpdateBasisAndGradientParticles(*grid, particles);
  
  // TODO(yiminlin.tri): Dp_inv_ is hardcoded for quadratic B-Spline
  // The values of Dp_inv_ are different for different B-Spline bases
  // https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf
  Dp_inv_ = 4.0 / (grid->get_h() * grid->get_h());
  particles->set_num_active_grid_nodes(grid->get_num_active_gridpt());
}

template <typename T>
int MPMTransfer<T>::MakeGridCompatibleWithParticles(Particles<T>* particles, SparseGrid<T>* grid) {
  int num_particles = particles->get_num_particles(); 
  std::vector<Vector3<int>> batch_indices(
      num_particles);  // batch_indices is the 3Dindex of batch that each particle belongs to
  // Preallocate the indices of batches
  for (int p = 0; p < num_particles; ++p) {
      batch_indices[p] = CalcBatchIndex(particles->get_position(p), grid->get_h());
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
  SortParticles(batch_indices, *grid, particles);  // sort particles based on sorted grid nodes above
  UpdateBasisAndGradientParticles(*grid, particles);
  particles->set_num_active_grid_nodes(grid->get_num_active_gridpt());
  return grid->get_num_active_gridpt(); 
}

template <typename T>
T MPMTransfer<T>::CalcTotalParticleEnergy(const Particles<T>& particles) {
  T energy = 0;
  for (int p = 0; p < particles.get_num_particles(); ++p) {
      T volp = particles.get_reference_volume(p);
      energy = energy + volp* particles.CalcParticleEnergyDensity(p, particles.get_elastic_deformation_gradient_new(p));
  }
  return energy;
}

template <typename T>
void MPMTransfer<T>::UpdateParticlesFnew(const std::array<BatchState, 27>& batch_states, T dt, int p, Particles<T>* particles) {
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

template <typename T>
void MPMTransfer<T>::CalcParticleFnewG2P(const SparseGrid<T>& grid, T dt, Particles<T>* particles) {
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

template <typename T>
void MPMTransfer<T>::CalcGridForceP2G(const Particles<T>& particles, SparseGrid<T>* grid) {
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

template <typename T>
void MPMTransfer<T>::CalcHessianP2G(Particles<T>* particles, SparseGrid<T>* grid, MatrixX<T>* hessian) {
    hessian->resize(grid->get_num_active_gridpt()*3, grid->get_num_active_gridpt()*3);
    hessian->setZero();
    particles->resize_stress_derivatives(particles->get_num_particles());
    particles->resize_stress_derivatives_contractF_contractF(particles->get_num_particles());
    particles->ComputePiolaDerivatives();
    particles->ContractPiolaDerivativesWithFWithF();
    int p_start, p_end;
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
                // loop over each i of 27 neighboring nodes in the numerator and each dimension α
                for (size_t i = 0; i < 27; ++i){ for (size_t alpha = 0; alpha < 3; ++alpha) {
                    // loop over each j of 27 neighboring nodes in the denominator and each dimension rho
                    for (size_t j = 0; j < 27; ++j){ for (size_t rho = 0; rho < 3; ++rho){
                        const Vector3<T>& gradNi_p = bases_grad_particles_[p][i];
                        const Vector3<T>& gradNj_p = bases_grad_particles_[p][j];
                        Eigen::Matrix<T, 9, 9>& A_alphabeta_rhogamma = particles->get_stress_derivatives_contractF_contractF_(p); //A(3β+α, 3γ+ρ)
                        // compute ∑ᵦᵧ A(3β+α, 3γ+ρ) ⋅ ∇Nᵢ(xₚ)[β] ⋅ ∇Nⱼ(xₚ)[γ]
                        for (size_t beta = 0; beta<3; ++beta) {
                            for (size_t gamma = 0; gamma<3; ++gamma) {
                                pad_hessian(3*i+alpha, 3*j+rho) += gradNi_p(beta) * A_alphabeta_rhogamma(beta*3+alpha, gamma*3+rho) * gradNj_p(gamma);
                            }
                        }
                    }}
                }}
            }
            p_start = p_end;            
            // assign local hessian to global, local hessian is (27*3) by (27*3)
            for (int ic = -1; ic <= 1; ++ic){ for (int ib = -1; ib <= 1; ++ib){ for (int ia = -1; ia <= 1; ++ia){
                int idx_local_i = (ia + 1) + 3 * (ib + 1) + 9 * (ic + 1); // local index 0-26
                Vector3<int> node_i_index_3d = batch_index_3d + Vector3<int>(ia, ib, ic); // global 3d index of node i

                for (int jc = -1; jc <= 1; ++jc){ for (int jb = -1; jb <= 1; ++jb){ for (int ja = -1; ja <= 1; ++ja){
                    int idx_local_j = (ja + 1) + 3 * (jb + 1) + 9 * (jc + 1); // local index 0-26
                    Vector3<int> node_j_index_3d = batch_index_3d + Vector3<int>(ja, jb, jc); // global 3d index of node i

                    MatrixX<T> dfi_dvj_block = pad_hessian.block(idx_local_i*3, idx_local_j*3, 3, 3); // get the local dfi_dvj block
                    (*hessian).block(grid->Reduce3DIndex(node_i_index_3d)*3, grid->Reduce3DIndex(node_j_index_3d)*3, 3, 3) += dfi_dvj_block;
                }}}
            }}}
        }
    }
}

template <typename T>
T MPMTransfer<T>::CalcEnergyForceHessian(Particles<T>* particles, SparseGrid<T>* grid, MatrixX<T>* hessian, T dt) {
    UpdateBasisAndGradientParticles(*grid, particles); // compute weights and weight gradients
    CalcParticleFnewG2P(*grid, dt, particles); //Fnew(vi*, dt) write to particles
    T energy = CalcTotalParticleEnergy(*particles);
    CalcGridForceP2G(*particles, grid); // write grid force and hessian onto grid
    CalcHessianP2G(particles, grid, hessian);
    return energy;
}

template <typename T>
void MPMTransfer<T>::TransferParticlesToGrid(const Particles<T>& particles,
                                             SparseGrid<T>* grid) {
  int p_start, p_end, idx_local;
  T mass_p, ref_volume_p;
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

      // Preallocate positions at grid points in the current batch on a
      // local array
      batch_index_3d = grid->Expand1DIndex(i);
      for (int c = -1; c <= 1; ++c) {
        for (int b = -1; b <= 1; ++b) {
          for (int a = -1; a <= 1; ++a) {
            idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
            batch_positions[idx_local] =
                grid->get_position(batch_index_3d + Vector3<int>(a, b, c));
          }
        }
      }

      // Clear local scratch pad
      for (auto& s : local_pads[i]) {
        s.reset();
      }

      // For each particle in the batch (Assume particles are sorted with
      // respect to the batch index), accmulate masses, momemtum, and
      // forces into grid points affected by the particle.
      for (int p = p_start; p < p_end; ++p) {
        mass_p = particles.get_mass(p);
        ref_volume_p = particles.get_reference_volume(p);
        // The affine matrix Cp = Bp * Dp^-1
        const Matrix3<T> C_p = particles.get_B_matrix(p) * Dp_inv_;

        // update force via Kirchhoff stress
        // AccumulateGridStatesOnBatchKirchhoff(p, mass_p, ref_volume_p,
        //                             particles.get_position(p),
        //                             particles.get_velocity(p),
        //                             C_p,
        //                             particles.get_kirchhoff_stress(p),
        //                             batch_positions, &local_pads[i]);

        // update force via first PK stress
        AccumulateGridStatesOnBatchFirstPK(
            p, mass_p, ref_volume_p, particles.get_position(p),
            particles.get_velocity(p), C_p,
            particles.get_elastic_deformation_gradient(p),
            particles.get_first_PK_stress(p), batch_positions, &local_pads[i]);
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

template <typename T>
void MPMTransfer<T>::TransferGridToParticles(const SparseGrid<T>& grid, T dt,
                                             Particles<T>* particles) {
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
          idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
          Vector3<int> index_3d = Vector3<int>(bi + a, bj + b, bk + c);
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

template <typename T>
void MPMTransfer<T>::SortParticles(
    const std::vector<Vector3<int>>& batch_indices, const SparseGrid<T>& grid,
    Particles<T>* particles) {
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
    if (batch_sizes_[i] > 0) {
      num_batches_++;
    }
  }

  // start_time = Clock::now();
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&grid, &batch_indices_1D](size_t i1, size_t i2) {
              return batch_indices_1D[i1] < batch_indices_1D[i2];
            });

  // Reorder the particles
  particles->Reorder(sorted_indices);
}

template <typename T>
void MPMTransfer<T>::UpdateBasisAndGradientParticles(
    const SparseGrid<T>& grid, Particles<T>* particles) {
  int num_particles, p_start, p_end;
  Vector3<int> batch_index_3d;
  num_particles = particles->get_num_particles();

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
      const Vector3<T>& xp = particles->get_position(p);
      EvalBasisOnBatch(p, xp, grid, batch_index_3d, particles);
    }
    p_start = p_end;
  }
}

template <typename T>
void MPMTransfer<T>::EvalBasisOnBatch(int p, const Vector3<T>& xp,
                                      const SparseGrid<T>& grid,
                                      const Vector3<int>& batch_index_3d,
                                      Particles<T>* particles) {
  int bi = batch_index_3d[0];
  int bj = batch_index_3d[1];
  int bk = batch_index_3d[2];
  int idx_local;
  Vector3<int> grid_index;

  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        grid_index(0) = bi + a;
        grid_index(1) = bj + b;
        grid_index(2) = bk + c;
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        // TODO(yiminlin.tri): Maybe a good idea to preallocate?
        BSpline<T> basis = BSpline<T>(grid.get_h(), grid.get_position(grid_index));
        std::tie(bases_val_particles_[p][idx_local],
                 bases_grad_particles_[p][idx_local]) =
            basis.EvalBasisAndGradient(xp);

        // also store the result in Particles
        particles->SetWeightAtParticle(p, idx_local, bases_val_particles_[p][idx_local]);
        particles->SetWeightGradientAtParticle(p, idx_local, bases_grad_particles_[p][idx_local]);
        particles->SetNeighborGridGlobalIndex(p, idx_local, grid.Reduce3DIndex(grid_index));
      }
    }
  }
}

template <typename T>
void MPMTransfer<T>::AccumulateGridStatesOnBatchKirchhoff(
    int p, T m_p, T reference_volume_p, const Vector3<T>& x_p,
    const Vector3<T>& v_p, const Matrix3<T>& C_p, const Matrix3<T>& tau_p,
    const std::array<Vector3<T>, 27>& batch_positions,
    std::array<GridState, 27>* local_pad) {
  int idx_local;
  T Ni_p;

  // Accumulate on local scratch pads
  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        Ni_p = bases_val_particles_[p][idx_local];
        const Vector3<T>& x_i = batch_positions[idx_local];
        const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        GridState& state_i = (*local_pad)[idx_local];
        T m_ip = m_p * Ni_p;
        state_i.mass += m_ip;
        // PIC update:
        // state_i.velocity += m_ip*v_p;
        // APIC update:
        state_i.velocity += m_ip * (v_p + C_p * (x_i - x_p));
        state_i.force += -reference_volume_p * tau_p * gradNi_p;
      }
    }
  }
}

template <typename T>
void MPMTransfer<T>::AccumulateGridStatesOnBatchFirstPK(
    int p, T m_p, T reference_volume_p, const Vector3<T>& x_p,
    const Vector3<T>& v_p, const Matrix3<T>& C_p, const Matrix3<T>& FE_p,
    const Matrix3<T>& P_p, const std::array<Vector3<T>, 27>& batch_positions,
    std::array<GridState, 27>* local_pad) {
  int idx_local;
  T Ni_p;

  // Accumulate on local scratch pads
  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        Ni_p = bases_val_particles_[p][idx_local];
        const Vector3<T>& x_i = batch_positions[idx_local];
        const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
        // For each particle in the batch (Assume particles are sorted with
        // respect to the batch index), update basis evaluations
        GridState& state_i = (*local_pad)[idx_local];
        T m_ip = m_p * Ni_p;
        state_i.mass += m_ip;
        // PIC update:
        // state_i.velocity += m_ip*v_p;
        // APIC update:
        state_i.velocity += m_ip * (v_p + C_p * (x_i - x_p));
        state_i.force +=
            -reference_volume_p * P_p * FE_p.transpose() * gradNi_p;
      }
    }
  }
}

//fᵢ += -V^0_p \cdot PKstress \cdot  F_p^T \cdot \nabla N_i(x_p)
template <typename T>
void MPMTransfer<T>::AccumulateGridForcesOnBatch(int p, T reference_volume_p, const Matrix3<T>& PK_stress, const Matrix3<T>& FE,
                                     std::array<GridState, 27>* local_pad) {
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

template <typename T>
void MPMTransfer<T>::WriteBatchStateToGrid(
    const Vector3<int>& batch_index_3d,
    const std::array<GridState, 27>& local_pad, SparseGrid<T>* grid) {
  int bi = batch_index_3d[0];
  int bj = batch_index_3d[1];
  int bk = batch_index_3d[2];
  int idx_local;
  Vector3<int> grid_index;

  // Put local scratch pad states into the grid
  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        grid_index(0) = bi + a;
        grid_index(1) = bj + b;
        grid_index(2) = bk + c;
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        const GridState& state_i = local_pad[idx_local];
        grid->AccumulateMass(grid_index, state_i.mass);
        grid->AccumulateVelocity(grid_index, state_i.velocity);
        grid->AccumulateForce(grid_index, state_i.force);
      }
    }
  }
}

template <typename T>
void MPMTransfer<T>::WriteBatchForceToGrid(
    const Vector3<int>& batch_index_3d,
    const std::array<GridState, 27>& local_pad, SparseGrid<T>* grid) {
  int bi = batch_index_3d[0];
  int bj = batch_index_3d[1];
  int bk = batch_index_3d[2];
  int idx_local;
  Vector3<int> grid_index;

  // Put local scratch pad states into the grid
  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        grid_index(0) = bi + a;
        grid_index(1) = bj + b;
        grid_index(2) = bk + c;
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        const GridState& state_i = local_pad[idx_local];
        grid->AccumulateForce(grid_index, state_i.force);
      }
    }
  }
}

template <typename T>
void MPMTransfer<T>::UpdateParticleStates(
    const std::array<BatchState, 27>& batch_states, T dt, int p,
    Particles<T>* particles) {
  int idx_local;
  T Ni_p;
  // vp_new_i = v_i^{n+1} * N_i(x_p)
  Vector3<T> vp_new_i;
  Vector3<T> xp = particles->get_position(p);

  // Scratch vectors and matrices
  Vector3<T> vp_new = Vector3<T>::Zero();
  Matrix3<T> Bp_new = Matrix3<T>::Zero();
  Matrix3<T> grad_vp_new = Matrix3<T>::Zero();

  // For each grid node affecting the current particle
  for (int c = -1; c <= 1; ++c) {
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        idx_local = (a + 1) + 3 * (b + 1) + 9 * (c + 1);
        Ni_p = bases_val_particles_[p][idx_local];
        const Vector3<T>& gradNi_p = bases_grad_particles_[p][idx_local];
        const Vector3<T>& xi = batch_states[idx_local].position;
        const Vector3<T>& vi_new = batch_states[idx_local].velocity;
        vp_new_i = vi_new * Ni_p;
        // v_p^{n+1} = \sum v_i^{n+1} N_i(x_p)
        vp_new += vp_new_i;
        // B_p^{n+1} = \sum v_i^{n+1}*(x_i - x_p^n)^T N_i(x_p)
        Bp_new += vp_new_i * (xi - xp).transpose();
        // Accumulate grad_vp_new: F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
        grad_vp_new += vi_new * gradNi_p.transpose();
      }
    }
  }

  // Update velocities and elastic deformation gradients of the particle
  // F_p^{n+1} = (I + dt*grad_vp_new)*F_p^n
  // Note that we assume the plastic deformation gradient doesn't change
  // during the evolution.
  particles->set_elastic_deformation_gradient(
      p, (Matrix3<T>::Identity() + dt * grad_vp_new) *
             particles->get_elastic_deformation_gradient(p));
  particles->set_velocity(p, vp_new);
  particles->set_B_matrix(p, Bp_new);
}

template <typename T>
Vector3<int> MPMTransfer<T>::CalcBatchIndex(const Vector3<T>& xp, T h) const {
  using std::round;
  return Vector3<int>(round(xp(0) / h), round(xp(1) / h), round(xp(2) / h));
}

template class MPMTransfer<double>;
template class MPMTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
