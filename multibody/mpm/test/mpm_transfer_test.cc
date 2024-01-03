#include "drake/multibody/mpm/mpm_transfer.h"

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {

using drake::multibody::mpm::constitutive_model::CorotatedElasticModel;
constexpr double kTolerance = 1e-12;

void CheckConservation(const internal::MassAndMomentum<double>& before,
                       const internal::MassAndMomentum<double>& after) {
  EXPECT_NEAR(before.total_mass, after.total_mass, kTolerance);
  EXPECT_TRUE(
      CompareMatrices(before.total_momentum, after.total_momentum, kTolerance));
  EXPECT_TRUE(CompareMatrices(before.total_angular_momentum,
                              after.total_angular_momentum, kTolerance));
}

// Randomly generate some particles
// Apply p2g
// Compare mass and momentum before and after
GTEST_TEST(MpmTransferTest, TestP2GAndG2PMassMomentumConservation) {
  const double h = 0.714285;

  SparseGrid<double> grid(h);
  Particles<double> particles = Particles<double>();

  size_t num_particles = 10;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-5.0, 5.0);
  std::uniform_real_distribution<double> positive_distribution(0.1, 2.0);

  CorotatedElasticModel<double> model(2.0, 0.2);

  for (size_t p = 0; p < num_particles; ++p) {
    Vector3<double> position{distribution(generator), distribution(generator),
                             distribution(generator)};
    Vector3<double> velocity{distribution(generator), distribution(generator),
                             distribution(generator)};
    double mass{positive_distribution(generator)};
    double volume{positive_distribution(generator)};

    Matrix3<double> B_matrix =
        (Matrix3<double>() << distribution(generator), distribution(generator),
         distribution(generator), distribution(generator),
         distribution(generator), distribution(generator),
         distribution(generator), distribution(generator),
         distribution(generator))
            .finished();

    particles.AddParticle(std::move(position), std::move(velocity),
                          model.Clone(), mass, volume);
    particles.SetBMatrixAt(p, std::move(B_matrix));
  }

  TransferScratch<double> scratch{};
  MpmTransfer<double> mpm_transfer{};
  GridData<double> grid_data{};
  ParticlesData<double> particles_data{};

  internal::MassAndMomentum<double> mass_and_momentum_from_particles =
      particles.ComputeTotalMassMomentum();

  mpm_transfer.SetUpTransfer(&grid, &particles);
  mpm_transfer.P2G(particles, grid, &grid_data, &scratch);

  internal::MassAndMomentum<double> mass_and_momentum_from_grid =
      grid.ComputeTotalMassMomentum(grid_data);

  CheckConservation(mass_and_momentum_from_particles,
                    mass_and_momentum_from_grid);

  // note: invoking p2g multiple times should still be valid
  mpm_transfer.P2G(particles, grid, &grid_data, &scratch);

  internal::MassAndMomentum<double> mass_and_momentum_from_grid2 =
      grid.ComputeTotalMassMomentum(grid_data);

  CheckConservation(mass_and_momentum_from_particles,
                    mass_and_momentum_from_grid2);

  // now we check G2P

  mpm_transfer.G2P(grid, grid_data, particles, &particles_data, &scratch);

  mpm_transfer.UpdateParticlesState(particles_data, 0.1, &particles);
  particles.AdvectParticles(0.1);

  internal::MassAndMomentum<double> mass_and_momentum_from_particles2 =
      particles.ComputeTotalMassMomentum();

  CheckConservation(mass_and_momentum_from_particles2,
                    mass_and_momentum_from_grid);

  // note: you cannot invoke G2P consecutively multiple times as particle
  // positions have changed.
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
