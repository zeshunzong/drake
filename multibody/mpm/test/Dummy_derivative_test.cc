#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"
#include <gtest/gtest.h>
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include <iostream>
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();

using Eigen::Matrix3d;



Matrix3<AutoDiffXd> MakeDeformationGradientsWithDerivatives() {
  /* Create an arbitrary AutoDiffXd deformation. */
  Matrix3d F;
  // clang-format off
  F << 0.18, 0.63, 0.54,
       0.13, 0.92, 0.17,
       0.03, 0.86, 0.85;
  // clang-format on
    const Eigen::Matrix<double, 9, Eigen::Dynamic> derivatives(
        Eigen::Matrix<double, 9, 9>::Identity());
    const auto F_autodiff_flat =
            math::InitializeAutoDiff(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(
                                        F.data(), 9),
                                    derivatives);
    Matrix3<AutoDiffXd> deformation_gradient_autodiff = Eigen::Map<const Matrix3<AutoDiffXd>>(F_autodiff_flat.data(), 3, 3);
    return deformation_gradient_autodiff;
}

void TestPIsDerivativeOfPsi(){
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();
    AutoDiffXd f = model.CalcStrainEnergyDensity(F);
    Matrix3<AutoDiffXd> dfdF;
    model.CalcFirstPiolaStress(F, &dfdF);
    const double kTolerance = 1e-12;
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3), dfdF,
         kTolerance));
}

void TestPIsDerivativeOfPsi2(){
    StvkHenckyWithVonMisesModel<AutoDiffXd> model(2.5, 0.3, 0.2);
    Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();
    AutoDiffXd f = model.CalcStrainEnergyDensity(F);
    Matrix3<AutoDiffXd> dfdF;
    model.CalcFirstPiolaStress(F, &dfdF);
    const double kTolerance = 1e-12;
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3), dfdF,
         kTolerance));
}

void TestTauIsPFT2(){
    //P Fᵀ = τ this should hold for first PK stress P and Kirchhoff stress τ
    StvkHenckyWithVonMisesModel<double> model(2.5, 0.3, 0.2);
    Matrix3d F;
    // clang-format off
    F << 0.18, 0.63, 0.54,
        0.13, 0.92, 0.17,
        0.03, 0.86, 0.85;
    // clang-format on
    Matrix3d tau;
    model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    // compute P F^T
    Matrix3d P;
    model.CalcFirstPiolaStress(F, &P);
    Matrix3d PFT = P * F.transpose();
    const double kTolerance = 1e-12;
    EXPECT_TRUE(CompareMatrices(
        tau, PFT,
         kTolerance));

}


void TestTauIsPFT(){
    //P Fᵀ = τ this should hold for first PK stress P and Kirchhoff stress τ
    CorotatedElasticModel<double> model(2.5,0.3);
    Matrix3d F;
    // clang-format off
    F << 0.18, 0.63, 0.54,
        0.13, 0.92, 0.17,
        0.03, 0.86, 0.85;
    // clang-format on
    Matrix3d tau;
    model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    // compute P F^T
    Matrix3d P;
    model.CalcFirstPiolaStress(F, &P);
    Matrix3d PFT = P * F.transpose();
    const double kTolerance = 1e-12;
    EXPECT_TRUE(CompareMatrices(
        tau, PFT,
         kTolerance));

}

void TestdPdFIsDerivativeOfP(){
    constexpr int kSpaceDimension = 3;
    const double kTolerance = 1e-12;
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();

    Matrix3<AutoDiffXd> P; 
    model.CalcFirstPiolaStress(F, &P);

    Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
    model.CalcFirstPiolaStressDerivative(F, &dPdF);
    for (int i = 0; i < kSpaceDimension; ++i) {
      for (int j = 0; j < kSpaceDimension; ++j) {
        Matrix3d dPijdF;
        for (int k = 0; k < kSpaceDimension; ++k) {
          for (int l = 0; l < kSpaceDimension; ++l) {
            dPijdF(k, l) = dPdF(3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(
            Eigen::Map<const Matrix3d>(P(i, j).derivatives().data(), 3, 3),
            dPijdF, kTolerance));
      }
    }

}

void TestdPdFIsDerivativeOfP2(){
    constexpr int kSpaceDimension = 3;
    const double kTolerance = 1e-12;
    StvkHenckyWithVonMisesModel<AutoDiffXd> model(2.5, 0.3, 0.2);
    Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();

    Matrix3<AutoDiffXd> P; 
    model.CalcFirstPiolaStress(F, &P);

    Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
    model.CalcFirstPiolaStressDerivative(F, &dPdF);
    for (int i = 0; i < kSpaceDimension; ++i) {
      for (int j = 0; j < kSpaceDimension; ++j) {
        Matrix3d dPijdF;
        for (int k = 0; k < kSpaceDimension; ++k) {
          for (int l = 0; l < kSpaceDimension; ++l) {
            dPijdF(k, l) = dPdF(3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(
            Eigen::Map<const Matrix3d>(P(i, j).derivatives().data(), 3, 3),
            dPijdF, kTolerance));
      }
    }

}

void TestDummyDerivative() {
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);

    Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();

    AutoDiffXd f = model.dummy_function(F);

    Matrix3<AutoDiffXd> dfdF;
    model.dummy_function_derivative(F, &dfdF);

    const double kTolerance = 1e-12;
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3), dfdF,
         kTolerance));

}



GTEST_TEST(Dummy_derivative_test, dummy_test) {
    TestDummyDerivative();

    TestPIsDerivativeOfPsi();
    TestdPdFIsDerivativeOfP();
    TestTauIsPFT();

    TestPIsDerivativeOfPsi2();
    TestTauIsPFT2();
    TestdPdFIsDerivativeOfP2();
}



}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
