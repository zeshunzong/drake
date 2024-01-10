#include "drake/multibody/mpm/mpm_driver.h"

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
constexpr double kTolerance = 1e-8;

GTEST_TEST(MpmDriverTest, FreeFallTest) {
  const double h = 0.197;
  double dt = 0.02;
  MpmDriver<double> driver = MpmDriver<double>(h, dt);
  // test free fall with three particles
  // give them same initial v so that there should be no deformation
  Vector3<double> initial_v(1.0, -2.0, 3.0);

  driver.AddParticle(Vector3<double>(0.1, 0.15, 0.07), initial_v,
                     std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2),
                     100.0, 100.0);
  driver.AddParticle(Vector3<double>(-0.48, -0.5, -0.5), initial_v,
                     std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2),
                     1.0, 1.0);
  driver.AddParticle(Vector3<double>(-0.48, -0.55, -0.5), initial_v,
                     std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2),
                     1.0, 1.0);

  driver.WriteParticlesToBgeo(0);
  for (int step = 1; step < 10; ++step) {
    // switch between matrix free or not
    if (step == 1) {
      driver.SetMatrixFree(false);
    }
    if (step == 5) {
      driver.SetMatrixFree(true);
    }
    driver.AdvanceDt();
    double current_time = step * dt;
    const Particles<double>& particles_at_current_time = driver.particles();

    // v_t should be v0 + gt
    for (size_t p = 0; p < particles_at_current_time.num_particles(); ++p) {
      const Vector3<double> v_t = particles_at_current_time.velocities()[p];
      EXPECT_TRUE(CompareMatrices(
          v_t, initial_v + driver.mpm_model().gravity() * current_time,
          kTolerance));
    }

    driver.WriteParticlesToBgeo(step);
  }
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
