#pragma once

#include <memory>
#include <vector>

#include "drake/multibody/mpm/internal/poisson_disk.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/sparse_grid.h"
#include "drake/multibody/mpm/internal/analytic_level_set.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct MpmState {
  explicit MpmState(double h) : sparse_grid(h) { DRAKE_DEMAND(h > 0); }

  int AddParticlesViaPoissonDiskSampling(
      const internal::AnalyticLevelSet& level_set,
      const math::RigidTransform<double>& pose,
      const mpm::constitutive_model::ElastoPlasticModel<T>& elastoplastic_model,
      double common_density) {
    const std::array<Vector3<double>, 2> bounding_box =
        level_set.bounding_box();
    // TODO(zeshunzong): pass is input
    int min_num_particles_per_cell = 1;
    double sample_r =
        sparse_grid.h() / (std::cbrt(min_num_particles_per_cell) + 1);

    std::array<double, 3> xmin = {bounding_box[0][0], bounding_box[0][1],
                                  bounding_box[0][2]};
    std::array<double, 3> xmax = {bounding_box[1][0], bounding_box[1][1],
                                  bounding_box[1][2]};
    // Generate sample particles in the reference frame
    std::vector<Vector3<double>> particles_sample_positions =
        internal::PoissonDiskSampling(sample_r, xmin, xmax);

    // Pick out sampled particles that are in the object
    int num_samples = particles_sample_positions.size();
    std::vector<Vector3<double>> particles_positions;
    for (int p = 0; p < num_samples; ++p) {
      // If the particle is in the level set, append its post-transfromed
      // position
      if (level_set.IsInClosure(particles_sample_positions[p])) {
        particles_positions.emplace_back(pose * particles_sample_positions[p]);
      }
    }

    int num_particles = particles_positions.size();
    // We assume every particle have the same volume and mass and constitutive
    // model
    double reference_volume_p = level_set.volume() / num_particles;
    double mass_p = common_density * reference_volume_p;
    // we assume all particles start with zero velocity
    for (int p = 0; p < num_particles; ++p) {
      particles.AddParticle(particles_positions[p], Vector3<T>(0, 0, 0),
                             elastoplastic_model.Clone(), mass_p,
                             reference_volume_p);
    }
    return num_particles;  // return the number of particles added
  }


  Particles<T> particles{};
  SparseGrid<T> sparse_grid;
};

template <typename T>
struct TransferScratch {
  // scratch pads for transferring states from particles to grid nodes
  // there will be one pad for each particle
  std::vector<P2gPad<T>> p2g_pads{};
  // scratch pad for transferring states from grid nodes to particles
  G2pPad<T> g2p_pad{};
};


template <typename T>
struct MpmSolverScratch {

  std::vector<Vector3<T>> v_prev; // previous step grid v
  Eigen::VectorX<T> minus_dEdv;
  Eigen::VectorX<T> dG; // change to be reflected on grid data
  std::vector<size_t> nodes_collide_with_ground;
  MatrixX<T> d2Edv2;
  TransferScratch<T> transfer_scratch;
  ParticlesData<T> particles_data;

  std::vector<size_t> collision_nodes;
  
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
