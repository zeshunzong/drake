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
                        Matrix3<double>::Ones() * 1.0);
  // add particle #2
  particles.AddParticle(Vector3<double>{2, 2, 2}, Vector3<double>{2, 2, 2}, 2.0,
                        2.0, Matrix3<double>::Ones() * 2.0,
                        Matrix3<double>::Ones() * 2.0);
  // add particle #3
  particles.AddParticle(Vector3<double>{3, 3, 3}, Vector3<double>{3, 3, 3}, 3.0,
                        3.0, Matrix3<double>::Ones() * 3.0,
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
  EXPECT_EQ(particles.deformation_gradients(), truth_matrix);
  EXPECT_EQ(particles.B_matrices(), truth_matrix);

  // getters for a single particle
  for (size_t p = 0; p < 3; p++) {
    Vector3<double> vec{p + 1.0, p + 1.0, p + 1.0};
    EXPECT_EQ(particles.GetPositionAt(p), vec);
    EXPECT_EQ(particles.GetVelocityAt(p), vec);
    EXPECT_EQ(particles.GetMassAt(p), p + 1.0);
    EXPECT_EQ(particles.GetReferenceVolumeAt(p), p + 1.0);
    Matrix3<double> mat = Matrix3<double>::Ones() * (p + 1.0);
    EXPECT_EQ(particles.GetDeformationGradientAt(p), mat);
    EXPECT_EQ(particles.GetBMatrixAt(p), mat);
  }
}

std::vector<std::vector<size_t>> GenerateAllPermutations(size_t n) {
  std::vector<size_t> base;
  for (size_t i = 0; i < n; ++i) {
    base.push_back(i);
  }
  // base = [0,1,2,...,n-1]
  size_t n_factorial = 1;
  for (size_t i = 1; i <= n; ++i) {
    n_factorial *= i;
  }
  std::vector<std::vector<size_t>> result;
  result.push_back(base);
  for (size_t i = 0; i < n_factorial - 1; ++i) {
    if (std::next_permutation(base.begin(), base.end())) {
      std::vector<size_t> next_perm = base;
      result.push_back(next_perm);
    } else {
      DRAKE_UNREACHABLE();
    }
  }
  return result;
}

void CheckReorder(const Particles<double>& particles,
                  const std::vector<size_t>& perm) {
  // assume the input particles has n particles, initialized with data
  // [1,2,3,...,n]
  Particles<double> particles_new = particles;
  particles_new.Reorder(perm);

  for (size_t i = 0; i < particles_new.num_particles(); i++) {
    // the i_th now particle is the original perm[i]'s particle,
    // its scaler field is (perm[i]+1)
    EXPECT_EQ(particles_new.GetMassAt(i), perm[i] + 1.0);
    // its vector field is vec3((perm[i]+1),(perm[i]+1),(perm[i]+1))
    EXPECT_EQ(particles_new.GetVelocityAt(i),
              Vector3<double>(perm[i] + 1.0, perm[i] + 1.0, perm[i] + 1.0));
    // its matrix field is Matrix3<double>::Ones() * (p + 1.0);
    EXPECT_EQ(particles_new.GetDeformationGradientAt(i),
              Matrix3<double>::Ones() * (perm[i] + 1.0));
  }
}

GTEST_TEST(ParticlesClassTest, TestReorder) {
  // test on 5 particles
  Particles<double> particles = Particles<double>();

  // add particle #1
  particles.AddParticle(Vector3<double>{1, 1, 1}, Vector3<double>{1, 1, 1}, 1.0,
                        1.0, Matrix3<double>::Ones() * 1.0,
                        Matrix3<double>::Ones() * 1.0);
  // add particle #2
  particles.AddParticle(Vector3<double>{2, 2, 2}, Vector3<double>{2, 2, 2}, 2.0,
                        2.0, Matrix3<double>::Ones() * 2.0,
                        Matrix3<double>::Ones() * 2.0);
  // add particle #3
  particles.AddParticle(Vector3<double>{3, 3, 3}, Vector3<double>{3, 3, 3}, 3.0,
                        3.0, Matrix3<double>::Ones() * 3.0,
                        Matrix3<double>::Ones() * 3.0);
  // add particle #4
  particles.AddParticle(Vector3<double>{4, 4, 4}, Vector3<double>{4, 4, 4}, 4.0,
                        4.0, Matrix3<double>::Ones() * 4.0,
                        Matrix3<double>::Ones() * 4.0);
  // add particle #5
  particles.AddParticle(Vector3<double>{5, 5, 5}, Vector3<double>{5, 5, 5}, 5.0,
                        5.0, Matrix3<double>::Ones() * 5.0,
                        Matrix3<double>::Ones() * 5.0);

  EXPECT_EQ(particles.num_particles(), 5);

  // generate all 5! permutations and test each of them
  std::vector<std::vector<size_t>> all_permutations =
      GenerateAllPermutations(5);
  EXPECT_EQ(all_permutations.size(), 120);

  for (size_t i = 0; i < all_permutations.size(); i++) {
    CheckReorder(particles, all_permutations[i]);
  }
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
