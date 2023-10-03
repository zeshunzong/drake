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
  std::vector<BSpline<double>> bspline_array(kNumGridPts);

  // Construct bspline_array to hold all basis functions on the grid points
  for (int idx = 0; idx < kNumGridPts; ++idx) {
    i = idx % kNumGridPts1d;
    j = (idx % kNumGridPts2d) / kNumGridPts1d;
    k = idx / kNumGridPts2d;
    xi = -2.0 + i * h;
    yi = -2.0 + j * h;
    zi = -2.0 + k * h;
    bspline_array[idx] = BSpline<double>(h, Vector3<double>(xi, yi, zi));
  }

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
          sum_sample += bspline_array[idx].ComputeValue(sample_point);
          sum_gradient_sample +=
              bspline_array[idx].ComputeGradient(sample_point);
          std::tie(sample_val, sample_gradient) =
              bspline_array[idx].ComputeValueAndGradient(sample_point);
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

  // Construct bspline_array to hold all basis functions on the grid points
  for (int idx = 0; idx < kNumGridPts; ++idx) {
    i = idx % kNumGridPts1d;
    j = (idx % kNumGridPts2d) / kNumGridPts1d;
    k = idx / kNumGridPts2d;
    xi = i * h;
    yi = j * h;
    zi = k * h;
    bspline_array[idx] = BSpline<double>(h, Vector3<double>(xi, yi, zi));
  }

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
          sum_sample += bspline_array[idx].ComputeValue(sample_point);
          sum_gradient_sample +=
              bspline_array[idx].ComputeGradient(sample_point);
          std::tie(sample_val, sample_gradient) =
              bspline_array[idx].ComputeValueAndGradient(sample_point);
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

GTEST_TEST(BSplineTest, TestDerivativeRelation) {
  // Compare autodiff and gradient at three different locations
  Vector3<double> center = Vector3<double>{-1.7, 0.233, 4.123};
  double h = 3.14159;
  BSpline<AutoDiffXd> bs_ad(h, center);
  BSpline<double> bs(h, center);

  Vector3<double> point1d{-1.0, 0.233, 4.123};
  Vector3<AutoDiffXd> point1_ad = math::InitializeAutoDiff(point1d);
  AutoDiffXd value1 = bs_ad.ComputeValue(point1_ad);
  Vector3<double> grad_computed1 = bs.ComputeGradient(point1d);
  Vector3<double> grad_autodiff1 = value1.derivatives();
  EXPECT_EQ(grad_computed1, grad_autodiff1);

  Vector3<double> point2d{1.0, 0.233, 4.123};
  Vector3<AutoDiffXd> point2_ad = math::InitializeAutoDiff(point2d);
  AutoDiffXd value2 = bs_ad.ComputeValue(point2_ad);
  Vector3<double> grad_computed2 = bs.ComputeGradient(point2d);
  Vector3<double> grad_autodiff2 = value2.derivatives();
  EXPECT_EQ(grad_computed2, grad_autodiff2);

  Vector3<double> point3d{41.0, 0.233, 4.123};
  Vector3<AutoDiffXd> point3_ad = math::InitializeAutoDiff(point3d);
  AutoDiffXd value3 = bs_ad.ComputeValue(point3_ad);
  Vector3<double> grad_computed3 = bs.ComputeGradient(point3d);
  Vector3<double> grad_autodiff3 = value3.derivatives();
  EXPECT_EQ(grad_computed3, grad_autodiff3);

  // AutoDiffXd y = 1.1;
  // auto y_ad = math::InitializeAutoDiff(y)
  // AutoDiffXd z = bs.dummy(y_ad);
  // Eigen::VectorX<double> xxxx = (z.derivatives());
  // std::cout <<  xxxx(0)<< std::endl;
  // EXPECT_TRUE(1==0);

  // XPECT_EQ(der , grad1);
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
