#include "drake/multibody/mpm/internal/b_spline.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kTolerance = 1e-10;

GTEST_TEST(BSplineTest, TestGetterSetter) {
    Vector3<double> center1 = Vector3<double>{1.0, 2.0, 3.0};
    double h1 = 1.0;
    BSpline<double> bs1(h1, center1);
    EXPECT_EQ(bs1.get_h(), h1);
    EXPECT_TRUE(CompareMatrices(bs1.get_position(), center1,
                kTolerance));

    Vector3<double> center2 = Vector3<double>{-1.0, 2.0, -3.0};
    double h2 = 10.0;
    BSpline<double> bs2(h2, center2);
    EXPECT_EQ(bs2.get_h(), h2);
    EXPECT_TRUE(CompareMatrices(bs2.get_position(), center2,
                kTolerance));
}

GTEST_TEST(BSplineTest, TestInSupport) {
    Vector3<double> center = { 0.5,  1.0, -0.5};
    // Construct a reference 3D test basis, and test the InSupport
    BSpline bs1 = BSpline<double>();
    // Construct a 3D test basis on reference element, and test the InSupport
    BSpline bs2 = BSpline<double>(0.5, center);
    BSpline bs3 = BSpline<double>(2.0, center);
    Vector3<double> pos1  = { 0.0,  0.0,  0.0};
    Vector3<double> pos2  = { 1.0,  0.0,  0.0};
    Vector3<double> pos3  = { 0.0,  1.0,  0.0};
    Vector3<double> pos4  = { 0.0,  0.0,  1.0};
    Vector3<double> pos5  = { 1.0,  1.0,  1.0};
    Vector3<double> pos6  = {-1.0, -1.0, -1.0};
    Vector3<double> pos7  = { 2.0,  2.0,  2.0};
    Vector3<double> pos8  = {-2.0, -2.0, -2.0};
    Vector3<double> pos9  = {-1.5,  1.5, -1.5};
    Vector3<double> pos10 = { 1.0,  1.0, -4.0};
    Vector3<double> pos11 = { 4.0,  4.0, -4.0};
    Vector3<double> pos12 = { 4.0,  0.0, -4.0};

    ASSERT_TRUE(bs1.InSupport(pos1));
    ASSERT_TRUE(bs1.InSupport(pos2));
    ASSERT_TRUE(bs1.InSupport(pos3));
    ASSERT_TRUE(bs1.InSupport(pos4));
    ASSERT_TRUE(bs1.InSupport(pos5));
    ASSERT_TRUE(bs1.InSupport(pos6));
    ASSERT_TRUE(!bs1.InSupport(pos7));
    ASSERT_TRUE(!bs1.InSupport(pos8));
    ASSERT_TRUE(!bs1.InSupport(pos9));
    ASSERT_TRUE(!bs1.InSupport(pos10));
    ASSERT_TRUE(!bs1.InSupport(pos11));
    ASSERT_TRUE(!bs1.InSupport(pos12));

    ASSERT_TRUE(!bs2.InSupport(pos1));
    ASSERT_TRUE(!bs2.InSupport(pos2));
    ASSERT_TRUE(bs2.InSupport(pos3));
    ASSERT_TRUE(!bs2.InSupport(pos4));
    ASSERT_TRUE(!bs2.InSupport(pos5));
    ASSERT_TRUE(!bs2.InSupport(pos6));
    ASSERT_TRUE(!bs2.InSupport(pos7));
    ASSERT_TRUE(!bs2.InSupport(pos8));
    ASSERT_TRUE(!bs2.InSupport(pos9));
    ASSERT_TRUE(!bs2.InSupport(pos10));
    ASSERT_TRUE(!bs2.InSupport(pos11));
    ASSERT_TRUE(!bs2.InSupport(pos12));

    ASSERT_TRUE(bs3.InSupport(pos1));
    ASSERT_TRUE(bs3.InSupport(pos2));
    ASSERT_TRUE(bs3.InSupport(pos3));
    ASSERT_TRUE(bs3.InSupport(pos4));
    ASSERT_TRUE(bs3.InSupport(pos5));
    ASSERT_TRUE(bs3.InSupport(pos6));
    ASSERT_TRUE(bs3.InSupport(pos7));
    ASSERT_TRUE(!bs3.InSupport(pos8));
    ASSERT_TRUE(bs3.InSupport(pos9));
    ASSERT_TRUE(!bs3.InSupport(pos10));
    ASSERT_TRUE(!bs3.InSupport(pos11));
    ASSERT_TRUE(!bs3.InSupport(pos12));
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

    const int kNUM_GRID_PT    = 125;  // total number of grid points
    const int kNUM_GRID_PT_2D = 25;   // number of grid points in 2D
    const int kNUM_GRID_PT_1D = 5;    // number of grid points in each direction
    int i, j, k;
    double xi, yi, zi;
    double sum_sample, sum_sample2, sample_interval, sample_val;
    Vector3<double> sum_gradient_sample, sum_gradient_sample2, sample_gradient;
    double h = 1.0;
    Vector3<double> sample_point;
    std::vector<BSpline<double>> bspline_array(kNUM_GRID_PT);

    // Construct bspline_array to hold all basis functions on the grid points
    for (int idx = 0; idx < kNUM_GRID_PT; ++idx) {
        i = idx % kNUM_GRID_PT_1D;
        j = (idx % kNUM_GRID_PT_2D) / kNUM_GRID_PT_1D;
        k = idx / kNUM_GRID_PT_2D;
        xi = -2.0 + i*h;
        yi = -2.0 + j*h;
        zi = -2.0 + k*h;
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
        sum_sample  = 0.0;
        sum_sample2 = 0.0;
        sum_gradient_sample  = {0.0, 0.0, 0.0};
        sum_gradient_sample2 = {0.0, 0.0, 0.0};
        sample_point = {xi, yi, zi};
        for (int idx = 0; idx < kNUM_GRID_PT; ++idx) {
            i = idx % kNUM_GRID_PT_1D;
            j = (idx % kNUM_GRID_PT_2D) / kNUM_GRID_PT_1D;
            k = idx / kNUM_GRID_PT_2D;
            sum_sample += bspline_array[idx].EvalBasis(sample_point);
            sum_gradient_sample += bspline_array[idx].EvalGradientBasis(sample_point);
            std::tie(sample_val, sample_gradient) =
                bspline_array[idx].EvalBasisAndGradient(sample_point);
            sum_sample2 += sample_val;
            sum_gradient_sample2 += sample_gradient;
        }

        EXPECT_NEAR(sum_sample, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(sum_gradient_sample,
                                    Vector3<double>{0.0, 0.0, 0.0},
                                    kTolerance));
        EXPECT_NEAR(sum_sample2, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(sum_gradient_sample2,
                                    Vector3<double>{0.0, 0.0, 0.0},
                                    kTolerance));
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
    for (int idx = 0; idx < kNUM_GRID_PT; ++idx) {
        i = idx % kNUM_GRID_PT_1D;
        j = (idx % kNUM_GRID_PT_2D) / kNUM_GRID_PT_1D;
        k = idx / kNUM_GRID_PT_2D;
        xi = i*h;
        yi = j*h;
        zi = k*h;
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
        sum_sample  = 0.0;
        sum_sample2 = 0.0;
        sum_gradient_sample  = {0.0, 0.0, 0.0};
        sum_gradient_sample2 = {0.0, 0.0, 0.0};
        sample_point = {xi, yi, zi};
        for (int idx = 0; idx < kNUM_GRID_PT; ++idx) {
            i = idx % kNUM_GRID_PT_1D;
            j = (idx % kNUM_GRID_PT_2D) / kNUM_GRID_PT_1D;
            k = idx / kNUM_GRID_PT_2D;
            sum_sample += bspline_array[idx].EvalBasis(sample_point);
            sum_gradient_sample += bspline_array[idx].EvalGradientBasis(sample_point);
            std::tie(sample_val, sample_gradient) =
                bspline_array[idx].EvalBasisAndGradient(sample_point);
            sum_sample2 += sample_val;
            sum_gradient_sample2 += sample_gradient;
        }

        EXPECT_NEAR(sum_sample, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(sum_gradient_sample,
                                    Vector3<double>{0.0, 0.0, 0.0},
                                    kTolerance));
        EXPECT_NEAR(sum_sample2, 1.0, kTolerance);
        EXPECT_TRUE(CompareMatrices(sum_gradient_sample2,
                                    Vector3<double>{0.0, 0.0, 0.0},
                                    kTolerance));
    }
    }
    }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
