#include "drake/multibody/mpm/particles.h"

#include <algorithm>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace {

// TODO(zeshunzong): add more as more attributes come in
GTEST_TEST(ParticlesClassTest, TestAddSetGet) {
  Particles<double> particles = Particles<double>();

  // add particle #1
  particles.AddParticle(Vector3<double>{1, 1, 1}, Vector3<double>{1, 1, 1}, 1.0,
                        1.0, Matrix3<double>::Ones() * 1.0,
                        Matrix3<double>::Ones() * 1.0,
                        Matrix3<double>::Ones() * 1.0);
  // add particle #2
  particles.AddParticle(Vector3<double>{2, 2, 2}, Vector3<double>{2, 2, 2}, 2.0,
                        2.0, Matrix3<double>::Ones() * 2.0,
                        Matrix3<double>::Ones() * 2.0,
                        Matrix3<double>::Ones() * 2.0);
  // add particle #3
  particles.AddParticle(Vector3<double>{3, 3, 3}, Vector3<double>{3, 3, 3}, 3.0,
                        3.0, Matrix3<double>::Ones() * 3.0,
                        Matrix3<double>::Ones() * 3.0,
                        Matrix3<double>::Ones() * 3.0);

  EXPECT_EQ(particles.num_particles(), 3);

  // getters for all particles
  std::vector<Vector3<double>> truth_vec{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
  EXPECT_EQ(particles.positions(), truth_vec);
  EXPECT_EQ(particles.velocities(), truth_vec);
  std::vector<double> truth_scalar{1, 2, 3};
  EXPECT_EQ(particles.masses(), truth_scalar);
  EXPECT_EQ(particles.reference_volumes(), truth_scalar);
  std::vector<Matrix3<double>> truth_matrix{Matrix3<double>::Ones() * 1.0,
                                            Matrix3<double>::Ones() * 2.0,
                                            Matrix3<double>::Ones() * 3.0};
  EXPECT_EQ(particles.elastic_deformation_gradients(), truth_matrix);
  EXPECT_EQ(particles.trial_deformation_gradients(), truth_matrix);
  EXPECT_EQ(particles.B_matrices(), truth_matrix);

  // getters for a single particle
  for (size_t p = 0; p < 3; ++p) {
    Vector3<double> vec{p + 1.0, p + 1.0, p + 1.0};
    EXPECT_EQ(particles.GetPositionAt(p), vec);
    EXPECT_EQ(particles.GetVelocityAt(p), vec);
    EXPECT_EQ(particles.GetMassAt(p), p + 1.0);
    EXPECT_EQ(particles.GetReferenceVolumeAt(p), p + 1.0);
    Matrix3<double> mat = Matrix3<double>::Ones() * (p + 1.0);
    EXPECT_EQ(particles.GetElasticDeformationGradientAt(p), mat);
    EXPECT_EQ(particles.GetTrialDeformationGradientAt(p), mat);
    EXPECT_EQ(particles.GetBMatrixAt(p), mat);
  }
}

GTEST_TEST(ParticlesClassTest, TestParticlesBatch) {
  const double h = 1.0;

  Particles<double> particles = Particles<double>();

  Vector3<double> particle_A_position{2.1, 2.9, 3.9};
  // its batch index should be {2,3,4}

  Vector3<double> particle_B_position{2.9, 4.1, 5.1};
  // its batch index should be {3,4,5}

  Vector3<double> particle_C_position{-2.9, 0.7, -0.7};
  // its batch index should be {-3, 1, -1}

  Vector3<double> particle_D_position{2.3, 3.2, 4.1};
  // its batch index should be {2,3,4}

  Vector3<double> particle_E_position{1.7, 3.1, 4.4};
  // its batch index should be {2,3,4}

  particles.AddParticle(particle_A_position, Vector3<double>::Zero(), 1.0, 1.0);
  particles.AddParticle(particle_B_position, Vector3<double>::Zero(), 2.0, 2.0);
  particles.AddParticle(particle_C_position, Vector3<double>::Zero(), 3.0, 3.0);
  particles.AddParticle(particle_D_position, Vector3<double>::Zero(), 4.0, 4.0);
  particles.AddParticle(particle_E_position, Vector3<double>::Zero(), 5.0, 5.0);
  particles.Prepare(h);

  const std::vector<Vector3<int>> base_nodes = particles.base_nodes();
  EXPECT_EQ(base_nodes[0], Vector3<int>(-3, 1, -1));
  EXPECT_EQ(base_nodes[1], Vector3<int>(2, 3, 4));
  EXPECT_EQ(base_nodes[2], Vector3<int>(2, 3, 4));
  EXPECT_EQ(base_nodes[3], Vector3<int>(2, 3, 4));
  EXPECT_EQ(base_nodes[4], Vector3<int>(3, 4, 5));

  const std::vector<size_t> batch_starts = particles.batch_starts();
  EXPECT_EQ(batch_starts.size(), 3);
  EXPECT_EQ(batch_starts[0], 0);
  EXPECT_EQ(batch_starts[1], 1);
  EXPECT_EQ(batch_starts[2], 4);

  const std::vector<size_t> batch_sizes = particles.batch_sizes();
  EXPECT_EQ(batch_sizes.size(), 3);
  EXPECT_EQ(batch_sizes[0], 1);
  EXPECT_EQ(batch_sizes[1], 3);
  EXPECT_EQ(batch_sizes[2], 1);
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
