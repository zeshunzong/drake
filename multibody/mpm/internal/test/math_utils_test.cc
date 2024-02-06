#include "drake/multibody/mpm/internal/math_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {

constexpr double kTolerance = 1e-8;

GTEST_TEST(MathUtilsTest, LeviCivitaTest) {
  // Equals 1 for an even permutation
  EXPECT_EQ(internal::LeviCivita(0, 1, 2), 1);
  EXPECT_EQ(internal::LeviCivita(1, 2, 0), 1);
  EXPECT_EQ(internal::LeviCivita(2, 0, 1), 1);
  // Equals -1 for an odd permutation
  EXPECT_EQ(internal::LeviCivita(2, 1, 0), -1);
  EXPECT_EQ(internal::LeviCivita(1, 0, 2), -1);
  EXPECT_EQ(internal::LeviCivita(0, 2, 1), -1);
  // Equals 0 otherwise
  EXPECT_EQ(internal::LeviCivita(0, 1, 0), 0);
  EXPECT_EQ(internal::LeviCivita(2, 2, 0), 0);
  EXPECT_EQ(internal::LeviCivita(1, 2, 1), 0);
  EXPECT_EQ(internal::LeviCivita(1, 1, 2), 0);
}

GTEST_TEST(MathUtilsTest, APermutationTest) {
  const Matrix3<double> A =
      (Matrix3<double>() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
  EXPECT_TRUE(CompareMatrices(internal::ContractionWithLeviCivita<double>(A),
                              Vector3<double>(-2.0, 4.0, -2.0), kTolerance));
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

void TestClampAutoDiff(double x, double eps) {
  Vector1<AutoDiffXd> x_autodiff = math::InitializeAutoDiff(Vector1<double>(x));

  AutoDiffXd value = internal::ClampToEpsilon<AutoDiffXd>(x_autodiff(0), eps);
  Vector1<double> derivative = value.derivatives();
  if (std::abs(x) >= eps) {
    // when |x| is large enough, return x and derivative is 1
    EXPECT_EQ(derivative(0), 1.0);
    EXPECT_EQ(value.value(), x);
  } else {
    // when |x| is small, derivative is 0
    EXPECT_EQ(derivative(0), 0.0);
    // return sign(x)*eps, where sign(0):=1
    if (x >= 0) {
      EXPECT_EQ(value.value(), eps);
    } else {
      EXPECT_EQ(value.value(), -eps);
    }
  }
}

GTEST_TEST(MathUtilsTest, ClampDerivativeTest) {
  TestClampAutoDiff(1.2, 0.8);
  TestClampAutoDiff(0.5, 0.8);
  TestClampAutoDiff(0.0, 0.8);
  TestClampAutoDiff(-0.5, 0.8);
  TestClampAutoDiff(-1.2, 0.8);
}

GTEST_TEST(MathUtilsTest, LogTest) {
  double x_large = 2.17;    // when approximation is not used
  double x_small = 0.0047;  // when approximation is used

  // log(x+1)/x
  EXPECT_EQ(internal::CalcLogXPlus1OverX<double>(x_large),
            std::log(x_large + 1) / x_large);
  EXPECT_NEAR(internal::CalcLogXPlus1OverX<double>(x_small),
              std::log(x_small + 1) / x_small, kTolerance);

  // approximation is needed when x and y are close
  double y_far = x_large + 0.5;
  double y_close = x_large - 0.00047;

  // (logx-logy)/(x-y)
  EXPECT_NEAR(internal::CalcLogXMinusLogYOverXMinusY<double>(x_large, y_far),
              (std::log(x_large) - std::log(y_far)) / (x_large - y_far),
              kTolerance);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(internal::CalcLogXMinusLogYOverXMinusY<double>(x_large, y_close),
              (std::log(x_large) - std::log(y_close)) / (x_large - y_close),
              kTolerance);
  // approximation is used

  // (x logy - y logx)/(x-y)
  EXPECT_NEAR(internal::CalcXLogYMinusYLogXOverXMinusY<double>(x_large, y_far),
              (x_large * std::log(y_far) - y_far * std::log(x_large)) /
                  (x_large - y_far),
              kTolerance);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(
      internal::CalcXLogYMinusYLogXOverXMinusY<double>(x_large, y_close),
      (x_large * std::log(y_close) - y_close * std::log(x_large)) /
          (x_large - y_close),
      kTolerance);
  // approximation is used
}

GTEST_TEST(MathUtilsTest, ExpTest) {
  double x_large = 2.17;    // when approximation is not used
  double x_small = 0.0047;  // when approximation is used

  // (expx-1)/x
  EXPECT_EQ(internal::CalcExpXMinus1OverX<double>(x_large),
            (std::exp(x_large) - 1.0) / x_large);
  EXPECT_NEAR(internal::CalcExpXMinus1OverX<double>(x_small),
              (std::exp(x_small) - 1.0) / x_small, kTolerance);

  // (expx-expy)/(x-y)
  double y_far = x_large + 0.5;
  double y_close = x_large + 0.00047;

  EXPECT_NEAR(internal::CalcExpXMinusExpYOverXMinusY<double>(x_large, y_far),
              (std::exp(x_large) - std::exp(y_far)) / (x_large - y_far),
              kTolerance);
  // Although there is no approximation when x and y are far away from each
  // other, what we are computing is a mathematically equivalent form, thus not
  // exactly the same.
  EXPECT_NEAR(internal::CalcExpXMinusExpYOverXMinusY<double>(x_large, y_close),
              (std::exp(x_large) - std::exp(y_close)) / (x_large - y_close),
              kTolerance);
}

GTEST_TEST(MathUtilsTest, ProjectPDTest) {
  Eigen::Matrix3d V = Eigen::Matrix3d::Random(3, 3);
  V = 0.5 * (V + V.transpose());  // make is symmetric

  Eigen::EigenSolver<Eigen::Matrix3d> solver(V);
  Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
  // we know that there is one negative eigenvalue in the original matrix
  EXPECT_TRUE(eigenvalues(2) < 0);

  // now project it
  internal::MakePD<double>(&V);

  Eigen::EigenSolver<Eigen::Matrix3d> solver_after(V);
  Eigen::VectorXd eigenvalues_after = solver_after.eigenvalues().real();

  EXPECT_TRUE(eigenvalues_after(0) >= 0);
  EXPECT_TRUE(eigenvalues_after(1) >= 0);
  EXPECT_TRUE(eigenvalues_after(2) >= 0);
}

GTEST_TEST(MathUtilsTest, ProjectPD2DTest) {
  Eigen::Matrix2d V = Eigen::Matrix2d::Random(3, 3);
  V = 0.5 * (V + V.transpose());  // make is symmetric

  Eigen::EigenSolver<Eigen::Matrix2d> solver(V);
  Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
  // we know that there is one negative eigenvalue in the original matrix
  EXPECT_TRUE(eigenvalues(1) < 0);
  // now project it
  internal::MakePD2D<double>(&V);

  Eigen::EigenSolver<Eigen::Matrix2d> solver_after(V);
  Eigen::VectorXd eigenvalues_after = solver_after.eigenvalues().real();

  EXPECT_TRUE(eigenvalues_after(0) >= 0);
  EXPECT_TRUE(eigenvalues_after(1) >= 0);
}

GTEST_TEST(MathUtilsTest, SolveFTest) {
  Eigen::Matrix3d B = Eigen::Matrix3d::Random(3, 3);
  Eigen::Matrix3d B1 = (B.transpose() + B);  // make the input symmetric
  Eigen::Matrix3d R = Eigen::Matrix3d::Random(3, 3);
  Eigen::Matrix3d F_in_out = Eigen::Matrix3d::Random(3, 3);

  internal::SolveForNewF(B1, R, &F_in_out);
  Eigen::Matrix3d Q0, S0;
  fem::internal::PolarDecompose<double>(F_in_out, &Q0, &S0);

  EXPECT_TRUE(CompareMatrices(
      B1, 0.5 * (R.transpose() * F_in_out + F_in_out.transpose() * R),
      kTolerance));

  Eigen::Matrix3d Q1, S1;
  fem::internal::PolarDecompose<double>(F_in_out, &Q1, &S1);

  EXPECT_TRUE(CompareMatrices(Q0, Q1, kTolerance));
}

GTEST_TEST(MathUtilsTest, SolveFTest2) {
  Eigen::Matrix3d B = Eigen::Matrix3d::Random(3, 3);
  Eigen::Matrix3d B1 = (B.transpose() + B);  // make the input symmetric
  Eigen::Matrix3d R = Eigen::Matrix3d::Random(3, 3);
  Eigen::Matrix3d F_in_out = Eigen::Matrix3d::Random(3, 3);

  internal::SolveForNewF2(B1, R, &F_in_out);
  Eigen::Matrix3d Q0, S0;
  fem::internal::PolarDecompose<double>(F_in_out, &Q0, &S0);

  EXPECT_TRUE(CompareMatrices(
      B1, 0.5 * (F_in_out * R.transpose() + R * F_in_out.transpose()),
      kTolerance));

  Eigen::Matrix3d Q1, S1;
  fem::internal::PolarDecompose<double>(F_in_out, &Q1, &S1);

  EXPECT_TRUE(CompareMatrices(Q0, Q1, kTolerance));
}

GTEST_TEST(MathUtilsTest, ProjectToSkewedCylinderTest) {
  double radius = 1.34;
  Eigen::Vector3d point(1, 0, 0);
  internal::ProjectToSkewedCylinder(radius, &point);

  EXPECT_EQ(point, Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d point2(3, 4, 5);
  internal::ProjectToSkewedCylinder(radius, &point2);

  // vec_diff: point on cylinder -> original point
  Eigen::Vector3d vec_diff = point2 - Eigen::Vector3d(3, 4, 5);
  // this should be perpendicular to axis
  EXPECT_NEAR(vec_diff.dot(Eigen::Vector3d(1, 1, 1)), 0, kTolerance);

  // compute point-to-axis distance
  // A: the point
  // B and C: two points on axis, take 000, 111
  Eigen::Vector3d BA = point2;
  Eigen::Vector3d BC(1, 1, 1);

  EXPECT_NEAR(radius, (BA.cross(BC)).norm() / BC.norm(), kTolerance);
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
