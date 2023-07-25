#include "drake/multibody/fem/mpm-dev/MathUtils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double TOLERANCE = 1e-12;

GTEST_TEST(MathUtilsTest, LeviCivitaTest) {
    // Equals 1 for an even permutation
    EXPECT_EQ(mathutils::LeviCivita(0, 1, 2), 1);
    EXPECT_EQ(mathutils::LeviCivita(1, 2, 0), 1);
    EXPECT_EQ(mathutils::LeviCivita(2, 0, 1), 1);
    // Equals -1 for an odd permutation
    EXPECT_EQ(mathutils::LeviCivita(2, 1, 0), -1);
    EXPECT_EQ(mathutils::LeviCivita(1, 0, 2), -1);
    EXPECT_EQ(mathutils::LeviCivita(0, 2, 1), -1);
    // Equals 0 otherwise
    EXPECT_EQ(mathutils::LeviCivita(0, 1, 0), 0);
    EXPECT_EQ(mathutils::LeviCivita(2, 2, 0), 0);
    EXPECT_EQ(mathutils::LeviCivita(1, 2, 1), 0);
    EXPECT_EQ(mathutils::LeviCivita(1, 1, 2), 0);
}

GTEST_TEST(MathUtilsTest, APermutationTest) {
    const Matrix3<double> A = (Matrix3<double>() <<
            1, 2, 3,
            4, 5, 6,
            7, 8, 9).finished();
    EXPECT_TRUE(CompareMatrices(mathutils::ContractionWithLeviCivita(A),
                                Vector3<double>(-2.0, 4.0, -2.0),
                                TOLERANCE));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
