#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/SparseGrid.h"
#include "drake/multibody/mpm/MPMTransfer.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"
#include <gtest/gtest.h>
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include <iostream>
#include <vector>
#include "drake/multibody/plant/deformable_model.h"
#include <stdlib.h> 


namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;





Eigen::Matrix3X<AutoDiffXd> MakeXpWithDerivatives(){
    constexpr int kParticles=10;
    Eigen::Matrix3Xd Xp = Eigen::MatrixXd::Random(3,kParticles);

    const Eigen::Matrix<double, 3*kParticles, Eigen::Dynamic> derivatives(
        Eigen::Matrix<double, 3*kParticles, 3*kParticles>::Identity());
    const auto Xp_autodiff_flat =
        math::InitializeAutoDiff(Eigen::Map<const Eigen::Matrix<double, 3*kParticles, 1>>(
                                    Xp.data(), 3*kParticles),
                                derivatives);
    Eigen::Matrix3X<AutoDiffXd> Xp_autodiff = Eigen::Map<const Matrix3X<AutoDiffXd>>(Xp_autodiff_flat.data(), 3, kParticles);
    return Xp_autodiff;
}

Eigen::Matrix3X<AutoDiffXd> MakeMatrix3XWithDerivatives(){
    constexpr int columns=27;
    Eigen::Matrix3Xd X = Eigen::MatrixXd::Random(3,columns);

    const Eigen::Matrix<double, 3*columns, Eigen::Dynamic> derivatives(
        Eigen::Matrix<double, 3*columns, 3*columns>::Identity());
    const auto X_autodiff_flat =
        math::InitializeAutoDiff(Eigen::Map<const Eigen::Matrix<double, 3*columns, 1>>(
                                    X.data(), 3*columns),
                                derivatives);
    Eigen::Matrix3X<AutoDiffXd> X_autodiff = Eigen::Map<const Matrix3X<AutoDiffXd>>(X_autodiff_flat.data(), 3, columns);
    return X_autodiff;
}


void TestEnergyAndForce(){
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    Eigen::Matrix3X<AutoDiffXd> Xp = MakeXpWithDerivatives();
    AutoDiffXd f = model.dummy_function2(Xp);
    Eigen::Matrix3X<AutoDiffXd> dfdXp;
    model.dummy_function2_derivative(Xp, &dfdXp);
    const double kTolerance = 1e-12;
    // num particles = 10;
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Eigen::Matrix3Xd>(f.derivatives().data(), 3, 10), dfdXp,
         kTolerance));
}


void xx(){

    mpm::Particles<AutoDiffXd> particles(0);
    SparseGrid<AutoDiffXd> grid{0.6};
    MPMTransfer<AutoDiffXd> mpm_transfer{};
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p = model.Clone();

    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.5,0.5,0.5}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 1.0, 10.0,
            Eigen::Matrix3<AutoDiffXd>::Identity(), Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p));
    
    int num_active_grids = mpm_transfer.DerivativeTest1(&grid, &particles);
    EXPECT_EQ(num_active_grids, 27);

    Eigen::Matrix3X<AutoDiffXd> Vi = MakeMatrix3XWithDerivatives();
    std::vector<Eigen::Vector3<AutoDiffXd>> grid_velocities_input{};

    for (size_t i = 0; i < 27; i++){
        grid_velocities_input.push_back(Vi.col(i));
    }

    mpm_transfer.DerivativeTest2(&grid, grid_velocities_input);
    mpm_transfer.DerivativeTest3(grid, 0.02, &particles);

    // for (int i = 0; i < kParticles; i++) {
    //     Eigen::Matrix3<AutoDiffXd> F_of_X;
    //     model.dummy_assign_F(Xp.col(0), &F_of_X);

    //     std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p = model.Clone();
    //     particles.AddParticle(Xp.col(0), Eigen::Vector3<AutoDiffXd>{0,0,0}, 1.0, 1.0,
    //                           F_of_X, Eigen::Matrix3<AutoDiffXd>::Identity() ,Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p));
    // }
    


}

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

    // TestEnergyAndForce();


}

// GTEST_TEST(Dummy_derivative_test, energy_test) {
//         xx();
// }



}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
