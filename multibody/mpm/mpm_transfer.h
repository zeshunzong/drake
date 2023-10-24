#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/sparse_grid.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/pad.h"


namespace drake {
namespace multibody {
namespace mpm {

// A implementation of MPM's Particles to Grid (P2G) and Grid to Particles (G2P)
// operations
template <typename T>
class MPMTransfer { 
 public:
    MPMTransfer() {}

    // sort grid
    // sort particles
    // update w, dw
    void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles) const;


    // Transfer masses, velocities, and Kirchhoff stresses on the particles
    // to masses, velocities, and forces on the grid
    void P2G(const Particles<T>& particles, const SparseGrid<T>& grid, GridData<T>* grid_data) const {

        std::vector<Pad<T>> local_pads(grid->num_active_nodes());

        // for batch_i in all batches {

        //     for particle p in batch_i {
        
        //          particles.pad_splatters_[p].SplatToPad(particles.masses_[p], ..., local_pads[i])
        
        //     }
        // }

        // write local pads to grid data
    }

    // Transfer velocities on the grids to velocities and deformation
    // gradients on the particles
    void TransferGridToParticles(const SparseGrid<T>& grid, GridData<T>* grid_data, double dt,
                                 Particles<T>* particles) const;




    
 private:

    void SortParticles(const std::vector<Vector3<int>>& batch_indices, Particles<T>* particles) const;



  


};  // class MPMTransfer

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
