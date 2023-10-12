#include "drake/multibody/mpm/internal/poisson_disk.h"

#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kTolerance = 4 * std::numeric_limits<double>::epsilon();

GTEST_TEST(PoissonDiskTest, TestOnePoint) {
  const double r = 2;

  const std::array<double, 3> x_min{-0.1, -0.1, -0.1};
  const std::array<double, 3> x_max{-0.1 + 1, -0.1 + 1, -0.1 + 1};

  auto result = PoissonDiskSampling(r, x_min, x_max);

  // radius is larger than diagonal of bounding box, should return only one
  // point
  EXPECT_EQ(result.size(), 1);
  // this point should be inside
  EXPECT_TRUE(result[0](0) >= -0.1);
  EXPECT_TRUE(result[0](0) <= -0.1 + 1);

  EXPECT_TRUE(result[0](1) >= -0.1);
  EXPECT_TRUE(result[0](1) <= -0.1 + 1);

  EXPECT_TRUE(result[0](2) >= -0.1);
  EXPECT_TRUE(result[0](2) <= -0.1 + 1);
}

GTEST_TEST(PoissonDiskTest, TestDistance) {
  const double r = 0.3;

  const std::array<double, 3> x_min{-0.1, -0.1, -0.1};
  const std::array<double, 3> x_max{-0.1 + 1, -0.1 + 1, -0.1 + 1};

  auto result = PoissonDiskSampling(r, x_min, x_max);

  // every point generated should be inside
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_TRUE(result[i](0) >= -0.1);
    EXPECT_TRUE(result[i](0) <= -0.1 + 1);

    EXPECT_TRUE(result[i](1) >= -0.1);
    EXPECT_TRUE(result[i](1) <= -0.1 + 1);

    EXPECT_TRUE(result[i](2) >= -0.1);
    EXPECT_TRUE(result[i](2) <= -0.1 + 1);
  }

  EXPECT_TRUE(result.size() > 1);
  // indeed more than one point. in fact, 32 points

  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i; j < result.size(); ++j) {
      double distance_sq =
          (result[i](0) - result[j](0)) * (result[i](0) - result[j](0)) +
          (result[i](1) - result[j](1)) * (result[i](1) - result[j](1)) +
          (result[i](2) - result[j](2)) * (result[i](2) - result[j](2));
      EXPECT_TRUE(distance_sq >= r * r);
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
