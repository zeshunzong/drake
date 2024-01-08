#include "drake/multibody/mpm/mpm_driver.h"

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {
using drake::multibody::mpm::constitutive_model::CorotatedElasticModel;
constexpr double kTolerance = 1e-10;

GTEST_TEST(MpmDriverTest, test1) {
  const double h = 0.2;
  double dt = 0.1;

  MpmDriver<double> driver = MpmDriver<double>(h, dt);

  driver.AddParticle(
      Vector3<double>(0.0, 0.0, 0.0), Vector3<double>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2), 1.0, 1.0);

  driver.AdvanceDt();

  EXPECT_FALSE(true);
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
