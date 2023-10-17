#include "drake/multibody/mpm/constitutive_model/stvk_hencky_with_von_mises_model.h"

#include <stdlib.h>

#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

constexpr double kTolerance = 1e-12;

namespace {

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;

Matrix3<AutoDiffXd> MakeDeformationGradientsWithDerivatives() {
  /* Create an arbitrary AutoDiffXd deformation. */
  Matrix3d F;

  // clang-format off
  F << 1.18, 0.63, 0.54,
       0.13, 0.92, 0.17,
       0.03, 0.86, 0.85;
  // clang-format on
  const Eigen::Matrix<double, 9, Eigen::Dynamic> derivatives(
      Eigen::Matrix<double, 9, 9>::Identity());
  const auto F_autodiff_flat = math::InitializeAutoDiff(
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(F.data(), 9), derivatives);
  Matrix3<AutoDiffXd> deformation_gradient_autodiff =
      Eigen::Map<const Matrix3<AutoDiffXd>>(F_autodiff_flat.data(), 3, 3);
  return deformation_gradient_autodiff;
}

GTEST_TEST(StvkWithVonMisesModelTest, SanityCheck) {
  // Sanity check: If F_trial is a rotation matrix, then stresses are
  // zero

  const double E = 5.0;
  const double nu = 0.25;
  const double yield_stress = 1e6;
  const StvkHenckyWithVonMisesModel<double> hencky_model(E, nu, yield_stress);
  const Matrix3<double> F_trial =
      math::RotationMatrix<double>(math::RollPitchYaw<double>(1.0, 2.0, 3.0))
          .matrix();
  const Matrix3<double> FE_exact = F_trial;

  Matrix3<double> FE;
  hencky_model.CalcFEFromFtrial(F_trial, &FE);
  // Apply return mapping to F_trial, it should be that FE = FE_exact = F_trial
  EXPECT_TRUE(CompareMatrices(FE, FE_exact));

  // Compute Kirchhoff stress from FE
  Matrix3<double> tau_computed;
  hencky_model.CalcKirchhoffStress(FE, &tau_computed);
  EXPECT_TRUE(
      CompareMatrices(tau_computed, Matrix3<double>::Zero(), kTolerance));

  // Compute First Piola stress from FE
  Matrix3<double> piola_stress_computed;
  hencky_model.CalcFirstPiolaStress(FE, &piola_stress_computed);
  EXPECT_TRUE(CompareMatrices(piola_stress_computed, Matrix3<double>::Zero(),
                              kTolerance));
}

// Pick a F_trial that is indeed in the plastic region. Apply return mapping.
// The returned elastic stress should be on the yield surface.
GTEST_TEST(StvkWithVonMisesModelTest, TestReturnMapping) {
  // clang-format off
    Matrix3<double> F_trial =
    (Matrix3<double>() << 1.18, 0.68, 0.54,
                          0.13, 0.92, 0.17,
                          0.03, 0.86, 0.85).finished();
  // clang-format on

  const double E = 23.4;
  const double nu = 0.41;
  const double yield_stress = 1.0;
  const StvkHenckyWithVonMisesModel<double> hencky_model = {E, nu,
                                                            yield_stress};

  const StvkHenckyWithVonMisesModel<double>::StrainStressData ssd =
      hencky_model.ComputeStrainStressData(F_trial);
  const double yield = hencky_model.ComputeYieldFunction(ssd.deviatoric_tau);
  EXPECT_TRUE(yield > 0.0);  // it indeed yields

  // apply returnMap to get FE
  Matrix3<double> FE;
  hencky_model.CalcFEFromFtrial(F_trial, &FE);

  const StvkHenckyWithVonMisesModel<double>::StrainStressData ssd_after =
      hencky_model.ComputeStrainStressData(FE);
  const double yield_after =
      hencky_model.ComputeYieldFunction(ssd_after.deviatoric_tau);
  EXPECT_NEAR(yield_after, 0.0, kTolerance);  // should not yield

  // the deviatoric_tau before and after should be parallel
  EXPECT_NEAR(ssd_after.deviatoric_tau(0) / ssd.deviatoric_tau(0),
              ssd_after.deviatoric_tau(1) / ssd.deviatoric_tau(1), kTolerance);
  EXPECT_NEAR(ssd_after.deviatoric_tau(1) / ssd.deviatoric_tau(1),
              ssd_after.deviatoric_tau(2) / ssd.deviatoric_tau(2), kTolerance);
}

GTEST_TEST(StvkWithVonMisesModelTest, TestPsiTauP) {
  // For  F with SVD
  //       = U ∑ V^T =  [1     0     0   [3 0 0   [ √2/2 √2/2   0
  //                     0  √2/2 -√2/2    0 2 0    -√2/2 √2/2   0
  //                     0  √2/2  √2/2]   0 0 1]       0    0   1]
  //                 =  [3/√2  3/√2     0
  //                       -1     1 -1/√2
  //                       -1     1  1/√2]
  // Then P = U ∑_P V^T, where ∑_P = 2μ ∑^-1 ln(∑) + λtr(ln(∑))∑^-1
  // Then P = [1     0     0
  //           0  √2/2 -√2/2
  //           0  √2/2  √2/2]
  //          diag[2/3μln(3)+λ/3(ln(3)+ln(2)),
  //                  μln(2)+λ/2(ln(3)+ln(2)),
  //                           λ(ln(3)+ln(2))]
  //          [ √2/2 √2/2   0
  //           -√2/2 √2/2   0
  //              0    0   1]
  // and tau = P*F^T
  // clang-format off
  const Matrix3<double> F_trial =
  (Matrix3<double>() << 3/sqrt(2), 3/sqrt(2),        0.0,
                           -1,         1,        -1/sqrt(2),
                           -1,         1,         1/sqrt(2)).finished();
  Matrix3<double> U =
  (Matrix3<double>() <<    1.0,       0.0,        0.0,
                           0.0,    sqrt(2)/2, -sqrt(2)/2,
                           0.0,    sqrt(2)/2,  sqrt(2)/2).finished();
  Matrix3<double> V =
  (Matrix3<double>() << sqrt(2)/2, -sqrt(2)/2,       0.0,
                        sqrt(2)/2,  sqrt(2)/2,       0.0,
                           0.0,       0.0,           1.0).finished();
  // clang-format on

  const double E = 23.4;
  const double nu = 0.41;
  const double yield_stress = 2e6;
  const StvkHenckyWithVonMisesModel<double> hencky_model = {E, nu,
                                                            yield_stress};

  // compute P by hand
  const double sum_log_sigma = std::log(3) + std::log(2);
  const double mu = E / (2 * (1 + nu));
  const double lambda = E * nu / (1 + nu) / (1 - 2 * nu);
  const Vector3<double> P_principal = {
      2.0 / 3.0 * mu * std::log(3) + lambda / 3.0 * sum_log_sigma,
      mu * std::log(2) + lambda / 2.0 * sum_log_sigma, lambda * sum_log_sigma};
  const Matrix3<double> P_exact = U * P_principal.asDiagonal() * V.transpose();

  Matrix3<double> FE;
  hencky_model.CalcFEFromFtrial(F_trial, &FE);
  Matrix3<double> piola_stress_computed;
  hencky_model.CalcFirstPiolaStress(FE, &piola_stress_computed);
  EXPECT_TRUE(CompareMatrices(piola_stress_computed, P_exact, kTolerance));

  Matrix3<double> tau_computed;
  hencky_model.CalcKirchhoffStress(FE, &tau_computed);
  EXPECT_TRUE(
      CompareMatrices(tau_computed, P_exact * FE.transpose(), kTolerance));
}

GTEST_TEST(StvkWithVonMisesModelTest, TestEnergyDerivatives) {
  // Test P is derivative of Psi
  const StvkHenckyWithVonMisesModel<AutoDiffXd> model(3.14159, 0.3337, 1e6);
  Matrix3<AutoDiffXd> FE = MakeDeformationGradientsWithDerivatives();
  // differentiate w.r.t. elastic deformation gradient. Here we neglect return
  // mapping
  AutoDiffXd f = model.CalcStrainEnergyDensity(FE);
  Matrix3<AutoDiffXd> dfdF;
  model.CalcFirstPiolaStress(FE, &dfdF);
  EXPECT_TRUE(
      CompareMatrices(Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3),
                      dfdF, kTolerance));

  // Test dPdF is derivative of P
  Matrix3<AutoDiffXd> P;
  model.CalcFirstPiolaStress(FE, &P);

  Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
  model.CalcFirstPiolaStressDerivative(FE, &dPdF);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Matrix3d dPijdF;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          dPijdF(k, l) = dPdF(3 * j + i, 3 * l + k).value();
        }
      }
      EXPECT_TRUE(CompareMatrices(
          Eigen::Map<const Matrix3d>(P(i, j).derivatives().data(), 3, 3),
          dPijdF, kTolerance));
    }
  }
}

}  // namespace
}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
