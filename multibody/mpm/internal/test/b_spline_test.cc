#include "drake/multibody/mpm/internal/b_spline.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kTolerance = 4 * std::numeric_limits<double>::epsilon();

GTEST_TEST(BSplineTest, TestGetterSetter) {
  Vector3<double> center1 = Vector3<double>{1.0, 2.0, 3.0};
  double h1 = 1.0;
  BSpline<double> bs1(h1, center1);
  EXPECT_EQ(bs1.h(), h1);
  EXPECT_EQ(bs1.center(), center1);

  Vector3<double> center2 = Vector3<double>{-1.0, 2.0, -3.0};
  double h2 = 10.0;
  BSpline<double> bs2(h2, center2);
  EXPECT_EQ(bs2.h(), h2);
  EXPECT_EQ(bs2.center(), center2);
}

GTEST_TEST(BSplineTest, TestBasis) {
  // First, Consider h = 1
  // Construct the basis on the grid of 5x5x5 on [-2,2]^3
  //                 -2  -1  0   1   2
  //                 o - o - o - o - o
  //                 |   |   |   |   |
  //             o - o - o - o - o - o
  //             |   |   |   |   |   |
  //         o - o - o - o - o - o - o
  //         |   |   |   |   |   |   |
  //     o - o - o - o - o - o - o - o
  //     |   |   |   |   |   |   |   |
  // o - o - o - o - o - o - o - o - o
  // |   |   |   |   |   |   |   |   |
  // o - o - o - o - o - o - o - o - o
  // |   |   |   |   |   |   |   |
  // o - o - o - o - o - o - o - o
  // |   |   |   |   |   |   |
  // o - o - o - o - o - o - o
  // |   |   |   |   |
  // o - o - o - o - o

  const int kNumGridPts = 125;   // total number of grid points
  const int kNumGridPts2d = 25;  // number of grid points in 2D
  const int kNumGridPts1d = 5;   // number of grid points in each direction
  int i, j, k;
  double xi, yi, zi;
  double sum_sample, sum_sample2, sample_interval, sample_val;
  Vector3<double> sum_gradient_sample, sum_gradient_sample2, sample_gradient;
  double h = 1.0;
  Vector3<double> sample_point;

  // Iterate through all sampled points over [-1, 1]^3. Then all basis
  // evaluated at sampled points shall be 1.0 by the partition of unity
  // property of B-spline basis. The gradient of the sum of bases should
  // then be 0.
  sample_interval = 0.3;
  for (zi = -1.5; zi <= 1.5; zi += sample_interval) {
    for (yi = -1.5; yi <= 1.5; yi += sample_interval) {
      for (xi = -1.5; xi <= 1.5; xi += sample_interval) {
        // Iterate through all Bsplines, and accumulate values
        sum_sample = 0.0;
        sum_sample2 = 0.0;
        sum_gradient_sample = {0.0, 0.0, 0.0};
        sum_gradient_sample2 = {0.0, 0.0, 0.0};
        sample_point = {xi, yi, zi};
        for (int idx = 0; idx < kNumGridPts; ++idx) {
          i = idx % kNumGridPts1d;
          j = (idx % kNumGridPts2d) / kNumGridPts1d;
          k = idx / kNumGridPts2d;
          xi = -2.0 + i * h;
          yi = -2.0 + j * h;
          zi = -2.0 + k * h;
          BSpline<double> bspline(h, Vector3<double>(xi, yi, zi));
          sum_sample += bspline.ComputeValue(sample_point);
          sum_gradient_sample += bspline.ComputeGradient(sample_point);
          std::tie(sample_val, sample_gradient) =
              bspline.ComputeValueAndGradient(sample_point);
          sum_sample2 += sample_val;
          sum_gradient_sample2 += sample_gradient;
        }

        EXPECT_NEAR(sum_sample, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(
            sum_gradient_sample, Vector3<double>{0.0, 0.0, 0.0}, kTolerance));
        EXPECT_NEAR(sum_sample2, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(
            sum_gradient_sample2, Vector3<double>{0.0, 0.0, 0.0}, kTolerance));
      }
    }
  }

  // Next, Consider h = 0.5, basis centered at (1.0, 1.0, 1.0)
  // Construct the basis on the grid of 5x5x5 on [0,2]^3
  //                 0   0.5 1  1.5  2
  //                 o - o - o - o - o 0
  //                 |   |   |   |   |
  //             o - o - o - o - o - o 0.5
  //             |   |   |   |   |   |
  //         o - o - o - o - o - o - o 1
  //         |   |   |   |   |   |   |
  //     o - o - o - o - o - o - o - o 1.5
  //     |   |   |   |   |   |   |   |
  // o - o - o - o - o - o - o - o - o 2.0
  // |   |   |   |   |   |   |   |
  // o - o - o - o - o - o - o - o
  // |   |   |   |   |   |   |
  // o - o - o - o - o - o - o
  // |   |   |   |   |   |
  // o - o - o - o - o - o
  // |   |   |   |   |
  // o - o - o - o - o

  h = 0.5;

  // Iterate through all sampled points over [0.25, 1.75]^3. Then all basis
  // evaluated at sampled points shall be 1.0 by the partition of unity
  // property of B-spline basis. The gradient of the sum of bases should
  // then be 0.
  sample_interval = 0.3;
  for (zi = 0.25; zi <= 1.75; zi += sample_interval) {
    for (yi = 0.25; yi <= 1.75; yi += sample_interval) {
      for (xi = 0.25; xi <= 1.75; xi += sample_interval) {
        // Iterate through all Bsplines, and accumulate values
        sum_sample = 0.0;
        sum_sample2 = 0.0;
        sum_gradient_sample = {0.0, 0.0, 0.0};
        sum_gradient_sample2 = {0.0, 0.0, 0.0};
        sample_point = {xi, yi, zi};
        for (int idx = 0; idx < kNumGridPts; ++idx) {
          i = idx % kNumGridPts1d;
          j = (idx % kNumGridPts2d) / kNumGridPts1d;
          k = idx / kNumGridPts2d;
          xi = i * h;
          yi = j * h;
          zi = k * h;
          BSpline<double> bspline(h, Vector3<double>(xi, yi, zi));
          sum_sample += bspline.ComputeValue(sample_point);
          sum_gradient_sample += bspline.ComputeGradient(sample_point);
          std::tie(sample_val, sample_gradient) =
              bspline.ComputeValueAndGradient(sample_point);
          sum_sample2 += sample_val;
          sum_gradient_sample2 += sample_gradient;
        }

        EXPECT_NEAR(sum_sample, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(
            sum_gradient_sample, Vector3<double>{0.0, 0.0, 0.0}, kTolerance));
        EXPECT_NEAR(sum_sample2, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(
            sum_gradient_sample2, Vector3<double>{0.0, 0.0, 0.0}, kTolerance));
      }
    }
  }
}

void CompareBSplineGradientWithAutoDiff(const BSpline<AutoDiffXd>& bs_autodiff,
                                        const BSpline<double>& bs_double,
                                        const Vector3<double>& position) {
  Vector3<AutoDiffXd> position_autodiff = math::InitializeAutoDiff(position);
  AutoDiffXd value = bs_autodiff.ComputeValue(position_autodiff);
  Vector3<double> grad_computed = bs_double.ComputeGradient(position);
  Vector3<double> grad_autodiff = value.derivatives();
  EXPECT_EQ(grad_computed, grad_autodiff);
}

GTEST_TEST(BSplineTest, TestDerivativeRelation) {
  // Compare autodiff and gradient at three different locations
  Vector3<double> center = Vector3<double>{-1.7, 0.233, 4.123};
  double h = 3.14159;
  BSpline<AutoDiffXd> bs_ad(h, center);
  BSpline<double> bs(h, center);

  Vector3<double> point1{-1.0, 0.233, 4.123};
  CompareBSplineGradientWithAutoDiff(bs_ad, bs, point1);

  Vector3<double> point2{1.0, 0.233, 4.123};
  CompareBSplineGradientWithAutoDiff(bs_ad, bs, point2);

  Vector3<double> point3{41.0, 0.233, 4.123};
  CompareBSplineGradientWithAutoDiff(bs_ad, bs, point3);
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
