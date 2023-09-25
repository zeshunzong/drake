#include "drake/multibody/mpm/internal/math_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {

constexpr double TOLERANCE = 1e-12;

GTEST_TEST(MathUtilsTest, LeviCivitaTest) {
  // Equals 1 for an even permutation
  EXPECT_EQ(internal::LeviCivita<double>(0, 1, 2), 1);
  EXPECT_EQ(internal::LeviCivita<double>(1, 2, 0), 1);
  EXPECT_EQ(internal::LeviCivita<double>(2, 0, 1), 1);
  // Equals -1 for an odd permutation
  EXPECT_EQ(internal::LeviCivita<double>(2, 1, 0), -1);
  EXPECT_EQ(internal::LeviCivita<double>(1, 0, 2), -1);
  EXPECT_EQ(internal::LeviCivita<double>(0, 2, 1), -1);
  // Equals 0 otherwise
  EXPECT_EQ(internal::LeviCivita<double>(0, 1, 0), 0);
  EXPECT_EQ(internal::LeviCivita<double>(2, 2, 0), 0);
  EXPECT_EQ(internal::LeviCivita<double>(1, 2, 1), 0);
  EXPECT_EQ(internal::LeviCivita<double>(1, 1, 2), 0);
}

GTEST_TEST(MathUtilsTest, APermutationTest) {
  const Matrix3<double> A =
      (Matrix3<double>() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
  EXPECT_TRUE(CompareMatrices(internal::ContractionWithLeviCivita<double>(A),
                              Vector3<double>(-2.0, 4.0, -2.0), TOLERANCE));
}

GTEST_TEST(MathUtilsTest, ClampTest) {
  double eps = 0.01;
  EXPECT_EQ(internal::ClampToEpsilon<double>(0.5, eps),
            0.5);  // x = 0.5, x > eps, returns x
  EXPECT_EQ(internal::ClampToEpsilon<double>(0.005, eps),
            eps);  // x = 0.005, 0 < x < eps, returns eps
  EXPECT_EQ(internal::ClampToEpsilon<double>(-0.005, eps),
            -eps);  // x = -0.005, -eps < x < 0, returns -eps
  EXPECT_EQ(internal::ClampToEpsilon<double>(-0.5, eps),
            -0.5);  // x = -0.5, x < -eps, returns x
  EXPECT_EQ(internal::ClampToEpsilon<double>(0, eps),
            eps);  // x = 0, can be either +eps or -eps, we choose it to be +eps
}

GTEST_TEST(MathUtilsTest, LogTest) {
  double eps = 0.01;

  double x_large = 2.17;    // when approximation is not used
  double x_small = 0.0047;  // when approximation is used

  // log(x+1)/x
  EXPECT_EQ(internal::CalcLogXPlus1OverX<double>(x_large, eps),
            std::log(x_large + 1) / x_large);
  EXPECT_NEAR(internal::CalcLogXPlus1OverX<double>(x_small, eps),
              std::log(x_small + 1) / x_small, 1e-3);

  // approximation is needed when x and y are close
  double y_far = x_large + 0.5;
  double y_close = x_large - 0.00047;

  //(logx-logy)/(x-y)
  EXPECT_NEAR(
      internal::CalcLogXMinusLogYOverXMinusY<double>(x_large, y_far, eps),
      (std::log(x_large) - std::log(y_far)) / (x_large - y_far), 1e-10);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(
      internal::CalcLogXMinusLogYOverXMinusY<double>(x_large, y_close, eps),
      (std::log(x_large) - std::log(y_close)) / (x_large - y_close), 1e-3);
  // approximation is used

  // (x logy - y logx)/(x-y)
  EXPECT_NEAR(internal::CalcXLogYMinusYLogXOverXMinusY(x_large, y_far, eps),
              (x_large * std::log(y_far) - y_far * std::log(x_large)) /
                  (x_large - y_far),
              1e-10);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(internal::CalcXLogYMinusYLogXOverXMinusY(x_large, y_close, eps),
              (x_large * std::log(y_close) - y_close * std::log(x_large)) /
                  (x_large - y_close),
              1e-3);
  // approximation is used
}

GTEST_TEST(MathUtilsTest, ExpTest) {
  double eps = 0.01;

  double x_large = 2.17;    // when approximation is not used
  double x_small = 0.0047;  // when approximation is used

  // (expx-1)/x
  EXPECT_EQ(internal::CalcExpXMinus1OverX<double>(x_large, eps),
            (std::exp(x_large) - 1.0) / x_large);
  EXPECT_NEAR(internal::CalcExpXMinus1OverX<double>(x_small, eps),
              (std::exp(x_small) - 1.0) / x_small, 1e-3);

  //(expx-expy)/(x-y)
  double y_far = x_large + 0.5;
  double y_close = x_large + 0.00047;

  EXPECT_NEAR(
      internal::CalcExpXMinusExpYOverXMinusY<double>(x_large, y_far, eps),
      (std::exp(x_large) - std::exp(y_far)) / (x_large - y_far), 1e-10);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(
      internal::CalcExpXMinusExpYOverXMinusY<double>(x_large, y_close, eps),
      (std::exp(x_large) - std::exp(y_close)) / (x_large - y_close), 1e-3);
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
