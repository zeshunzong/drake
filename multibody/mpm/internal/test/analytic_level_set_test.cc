#include "drake/multibody/mpm/internal/analytic_level_set.h"

#include <math.h>

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kTolerance = 4 * std::numeric_limits<double>::epsilon();

GTEST_TEST(AnalyticLevelSetTest, HalfSpaceTest) {
  const Vector3<double> normal = {3.0, 2.0, 1.0};

  // Check IsInClosure
  HalfSpaceLevelSet half_space = HalfSpaceLevelSet(normal);
  EXPECT_EQ(half_space.volume(), std::numeric_limits<double>::infinity());
  EXPECT_TRUE(half_space.IsInClosure({0.0, 0.0, 0.0}));
  EXPECT_TRUE(half_space.IsInClosure({-2.5, -1.5, 0.0}));
  EXPECT_TRUE(half_space.IsInClosure({-3.0, -2.0, -1.0}));
  EXPECT_FALSE(half_space.IsInClosure({3.0, 2.0, 1.0}));
  EXPECT_FALSE(half_space.IsInClosure({1.0, 2.0, 3.0}));

  // Check normal
  Vector3<double> normal_outward = {3.0 / sqrt(14), 2.0 / sqrt(14),
                                    1.0 / sqrt(14)};

  EXPECT_EQ(half_space.GetNormal({0.0, 0.0, -0.1}), normal_outward);
  EXPECT_EQ(half_space.GetNormal({0.0, 0.0, 0.0}), normal_outward);
  EXPECT_THROW(half_space.GetNormal({4.0, 3.0, 3.0}), std::exception);

  // Check Bounding Box
  const std::array<Vector3<double>, 2>& bounding_box =
      half_space.bounding_box();
  EXPECT_TRUE(
      CompareMatrices(bounding_box[0],
                      Vector3<double>{-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()},
                      kTolerance));
  EXPECT_TRUE(
      CompareMatrices(bounding_box[1],
                      Vector3<double>{std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()},
                      kTolerance));
}

GTEST_TEST(AnalyticLevelSetTest, BoxTest) {
  const Vector3<double> xscale = {3.0, 2.0, 1.0};

  // Check IsInClosure
  BoxLevelSet box = BoxLevelSet(xscale);
  EXPECT_EQ(box.volume(), 48.0);
  EXPECT_TRUE(box.IsInClosure({0.0, 0.0, 0.0}));
  EXPECT_TRUE(box.IsInClosure({1.0, -1.0, 0.0}));
  EXPECT_TRUE(box.IsInClosure({-2.5, -1.5, 0.0}));
  EXPECT_TRUE(box.IsInClosure({-3.0, -2.0, -1.0}));
  EXPECT_FALSE(box.IsInClosure({1.0, 2.0, 3.0}));

  // Check GetNormal
  Vector3<double> outward_normal = {0.0, 0.0, 1.0};
  EXPECT_EQ(box.GetNormal({0.0, 0.0, 0.1}), outward_normal);

  outward_normal = {0.0, 0.0, -1.0};
  EXPECT_EQ(box.GetNormal({1.0, 1.0, -0.9}), outward_normal);

  outward_normal = {0.0, -1.0, 0.0};
  EXPECT_EQ(box.GetNormal({1.0, -1.8, 0.0}), outward_normal);

  outward_normal = {0.0, 1.0, 0.0};
  EXPECT_EQ(box.GetNormal({1.0, 1.7, 0.6}), outward_normal);

  outward_normal = {1.0, 0.0, 0.0};
  EXPECT_EQ(box.GetNormal({2.5, -1.0, 0.2}), outward_normal);

  outward_normal = {-1.0, 0.0, 0.0};
  EXPECT_EQ(box.GetNormal({-2.1, -0.2, 0.0}), outward_normal);

  outward_normal = {-1.0, 0.0, 0.0};
  EXPECT_EQ(box.GetNormal({-2.0, -0.2, 0.0}), outward_normal);

  EXPECT_THROW(box.GetNormal({4.0, 3.0, 3.0}), std::exception);

  // Check Bounding Box
  const std::array<Vector3<double>, 2>& bounding_box = box.bounding_box();
  Vector3<double> bounding = {-3.0, -2.0, -1.0};
  EXPECT_EQ(bounding_box[0], bounding);

  bounding = {3.0, 2.0, 1.0};
  EXPECT_EQ(bounding_box[1], bounding);
}

GTEST_TEST(AnalyticLevelSetTest, SphereTest) {
  double radius = 2.0;

  SphereLevelSet sphere = SphereLevelSet(radius);

  // Check IsInClosure
  EXPECT_NEAR(sphere.volume(), 32.0 / 3.0 * M_PI, kTolerance);
  EXPECT_TRUE(sphere.IsInClosure({0.0, 0.0, 0.0}));
  EXPECT_TRUE(sphere.IsInClosure({0.2, 0.0, 0.2}));
  EXPECT_TRUE(sphere.IsInClosure({-0.5, -0.6, 0.7}));
  EXPECT_FALSE(sphere.IsInClosure({3.0, 0.0, -1.0}));

  // Check normal
  EXPECT_TRUE(CompareMatrices(sphere.GetNormal({1.0, 0.0, 0.0}),
                              Vector3<double>{1.0, 0.0, 0.0}, kTolerance));
  EXPECT_TRUE(CompareMatrices(sphere.GetNormal({0.0, -1.0, 0.0}),
                              Vector3<double>{0.0, -1.0, 0.0}, kTolerance));
  EXPECT_TRUE(CompareMatrices(sphere.GetNormal({0.0, 0.0, 1.0}),
                              Vector3<double>{0.0, 0.0, 1.0}, kTolerance));
  EXPECT_THROW(sphere.GetNormal({3.0, 3.0, 2.0}), std::exception);

  // Check Bounding Box
  const std::array<Vector3<double>, 2>& bounding_box = sphere.bounding_box();
  EXPECT_TRUE(CompareMatrices(bounding_box[0],
                              Vector3<double>{-2.0, -2.0, -2.0}, kTolerance));
  EXPECT_TRUE(CompareMatrices(bounding_box[1], Vector3<double>{2.0, 2.0, 2.0},
                              kTolerance));
}

GTEST_TEST(AnalyticLevelSetTest, CylinderTest) {
  double radius = 2.0;
  double height = 0.5;

  CylinderLevelSet cylinder = CylinderLevelSet(height, radius);

  // Check IsInClosure
  EXPECT_NEAR(cylinder.volume(), 4.0 * M_PI, kTolerance);
  EXPECT_TRUE(cylinder.IsInClosure({0.0, 0.0, 0.0}));
  EXPECT_TRUE(cylinder.IsInClosure({0.2, 0.0, 0.2}));
  EXPECT_FALSE(cylinder.IsInClosure({-0.5, -0.6, 0.7}));
  EXPECT_FALSE(cylinder.IsInClosure({3.0, 0.0, -1.0}));

  // Check normal
  // for a point on top disk, normal is (0,0,1)
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({0.0, 0.0, 0.5}),
                              Vector3<double>{0.0, 0.0, 1.0}, kTolerance));
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({-1.0, 1.1, 0.5}),
                              Vector3<double>{0.0, 0.0, 1.0}, kTolerance));
  // for a point on bottom disk, normal is (0,0,-1)
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({0.0, 0.0, -0.5}),
                              Vector3<double>{0.0, 0.0, -1.0}, kTolerance));
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({-1.0, 1.1, -0.5}),
                              Vector3<double>{0.0, 0.0, -1.0}, kTolerance));

  // for a point closer to the side
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({1.4, -1.4, 0.2}),
                              .5 * Vector3<double>{sqrt(2), -sqrt(2), 0.0},
                              kTolerance));
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({1.4, -1.4, -0.2}),
                              .5 * Vector3<double>{sqrt(2), -sqrt(2), 0.0},
                              kTolerance));

  // for a point closer to top/bottom
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({1.4, -1.4, 0.499}),
                              Vector3<double>{0.0, 0.0, 1.0}, kTolerance));
  // for a point closer to top/bottom
  EXPECT_TRUE(CompareMatrices(cylinder.GetNormal({1.4, -1.4, -0.499}),
                              Vector3<double>{0.0, 0.0, -1.0}, kTolerance));

  // Check Bounding Box
  const std::array<Vector3<double>, 2>& bounding_box = cylinder.bounding_box();

  Vector3<double> bounding = {-2.0, -2.0, -0.5};
  EXPECT_EQ(bounding_box[0], bounding);

  bounding = {2.0, 2.0, 0.5};
  EXPECT_EQ(bounding_box[1], bounding);
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
