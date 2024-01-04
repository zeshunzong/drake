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

// say the velocity field is (2x, 2y, 0)
Vector3<double> GetVelocityField(const Vector3<double>& position) {
  Vector3<double> v;
  v(0) = 2.0 * position(0);
  v(1) = 2.0 * position(1);
  v(2) = 0.0;
  return v;
}

GTEST_TEST(MpmTransferTest, TestGradV) {
  const double h = 0.1;

  SparseGrid<double> grid(h);
  Particles<double> particles = Particles<double>();
  particles.AddParticle(
      Vector3<double>(0.5, 0.5, 0.5), Vector3<double>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2), 1.0, 1.0);
  TransferScratch<double> scratch{};
  MpmTransfer<double> mpm_transfer{};
  GridData<double> grid_data{};
  ParticlesData<double> particles_data{};
  mpm_transfer.SetUpTransfer(&grid, &particles);
  mpm_transfer.P2G(particles, grid, &grid_data, &scratch);

  // manually change grid v
  std::vector<Vector3<double>> test_v_field;
  for (int i = 0; i < 27; ++i) {
    const Vector3<double> position =
        internal::ComputePositionFromIndex3D(grid.To3DIndex(i), grid.h());
    test_v_field.emplace_back(GetVelocityField(position));
  }
  grid_data.SetVelocities(test_v_field);

  mpm_transfer.G2P(grid, grid_data, particles, &particles_data, &scratch);

  Matrix3<double> correct_grad_v = Matrix3<double>::Zero();
  correct_grad_v(0, 0) = 2.0;
  correct_grad_v(1, 1) = 2.0;
  EXPECT_TRUE(CompareMatrices(particles_data.particle_grad_v_next[0],
                              correct_grad_v, kTolerance));
  // note: for a linear velocity field (we use (2x, 2y, 0) here) the grad_v
  // computed should in theory be exact, except for tiny numerical errors.
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
