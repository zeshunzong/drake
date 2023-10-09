#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

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
namespace {

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;

constexpr double kTolerance = 1e-12;

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
  const auto F_autodiff_flat = math::InitializeAutoDiff(
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(F.data(), 9), derivatives);
  Matrix3<AutoDiffXd> deformation_gradient_autodiff =
      Eigen::Map<const Matrix3<AutoDiffXd>>(F_autodiff_flat.data(), 3, 3);
  return deformation_gradient_autodiff;
}

GTEST_TEST(CorotatedElasticModelTest, TestReturnMapAndStress) {
  // Taken from fem/test/corotated_model_data_test.cc
  /* We set deformation gradient as F = R*S where R is an arbitrary rotation
  matrix and S is an arbitrary sysmmetric positive definite matrix. */
  const Matrix3<double> R =
      math::RotationMatrix<double>(
          math::RollPitchYaw<double>(M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0))
          .matrix();
  // clang-format off
    const Matrix3<double> S = (Matrix3<double>() <<
            6, 1, 2,
            1, 4, 1,
            2, 1, 5).finished();
  // clang-format on

  /* ==================== Test #1 ================ */
  // tau_exact1 is the exact tau computed from F=R*S, under E = 2.0 and mu =
  // 0.0.
  // clang-format off
    const Matrix3<double> tau_exact1 = (Matrix3<double>() <<
             50, -42,  20,
            -42,  70, -22,
             20, -22,  28).finished();
  // clang-format on

  Matrix3<double> F_trial1 = R * S;
  Matrix3<double> FE_exact =
      F_trial1;  // under Corotated Model, FE_exact = F_trial

  // Test on E = 2.0, nu = 0.0
  CorotatedElasticModel<double> model1(2.0, 0.0);

  model1.CalcFEFromFtrial(&F_trial1);
  // Now F_trial1 is essentially FE_1, the elastic deformation gradient

  // Compute Kirchhoff stress from FE
  Matrix3<double> tau_computed1;
  model1.CalcKirchhoffStress(F_trial1, &tau_computed1);

  // Compute First Piola stress from FE
  Matrix3<double> piola_stress_computed1;
  model1.CalcFirstPiolaStress(F_trial1, &piola_stress_computed1);

  // verify the relationship between P and τ
  // P = τ * FE^{-T}, or P * FE^T = τ
  EXPECT_TRUE(CompareMatrices(piola_stress_computed1 * F_trial1.transpose(),
                              tau_computed1, kTolerance));

  EXPECT_TRUE(CompareMatrices(tau_computed1, tau_exact1, kTolerance));
  // Sanity check: no plasticity shall be applied
  // F_trial_1 now is FE_computed, after return mapping
  EXPECT_TRUE(CompareMatrices(F_trial1, FE_exact));

  /* ==================== Test #2 ================ */
  // tau_exact2 is the exact tau computed from F=R*S, under E = 5.0 and mu =
  // 0.25.
  // clang-format off
    const Matrix3<double> tau_exact2 = (Matrix3<double>() <<
             18724, -84,  40,
            -84,  18764, -44,
             40, -44,  18680).finished();
  // clang-format on

  // Test on E = 5.0, nu = 0.25
  CorotatedElasticModel<double> model2(5.0, 0.25);
  // recreate a new F_trial
  Matrix3<double> F_trial2 = R * S;
  FE_exact = F_trial2;  // under Corotated Model, FE_exact = F_trial

  model2.CalcFEFromFtrial(&F_trial2);
  // Now F_trial2 is essentially FE_2, the elastic deformation gradient

  // Compute Kirchhoff stress from FE
  Matrix3<double> tau_computed2;
  model2.CalcKirchhoffStress(F_trial2, &tau_computed2);

  // Compute First Piola stress from FE
  Matrix3<double> piola_stress_computed2;
  model2.CalcFirstPiolaStress(F_trial2, &piola_stress_computed2);

  // verify the relationship between P and τ
  // P = τ * FE^{-T}, or P * FE^T = τ
  EXPECT_TRUE(CompareMatrices(piola_stress_computed2 * F_trial2.transpose(),
                              tau_computed2, kTolerance));

  EXPECT_TRUE(CompareMatrices(tau_computed2, tau_exact2, kTolerance));
  // Sanity check: no plasticity shall be applied
  // F_trial2 now is FE_computed, after return mapping
  EXPECT_TRUE(CompareMatrices(F_trial2, FE_exact));

  /* ==================== Test #3 ================ */
  // If F is a rotation matrix, then stress is zero
  Matrix3<double> F_rot =
      math::RotationMatrix<double>(math::RollPitchYaw<double>(1.0, 2.0, 3.0))
          .matrix();
  model2.CalcFEFromFtrial(&F_rot);
  Matrix3<double> tau_computed_zero;
  model2.CalcKirchhoffStress(F_rot, &tau_computed_zero);
  EXPECT_TRUE(
      CompareMatrices(tau_computed_zero, Matrix3<double>::Zero(), kTolerance));

  // Try another F
  Matrix3<double> F_rot2 =
      math::RotationMatrix<double>(math::RollPitchYaw<double>(0.1, -2.4, 13.3))
          .matrix();
  model1.CalcFEFromFtrial(&F_rot2);
  Matrix3<double> tau_computed_zero2;
  model1.CalcKirchhoffStress(F_rot2, &tau_computed_zero2);
  EXPECT_TRUE(
      CompareMatrices(tau_computed_zero2, Matrix3<double>::Zero(), kTolerance));
}

GTEST_TEST(CorotatedElasticModelTest, TestEnergyDerivatives) {
  // Test P is derivative of Psi
  CorotatedElasticModel<AutoDiffXd> model(3.14159, 0.3337);
  Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();
  AutoDiffXd f = model.CalcStrainEnergyDensity(F);
  Matrix3<AutoDiffXd> dfdF;
  model.CalcFirstPiolaStress(F, &dfdF);
  EXPECT_TRUE(
      CompareMatrices(Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3),
                      dfdF, kTolerance));

  // Test dPdF is derivative of P
  Matrix3<AutoDiffXd> P;
  model.CalcFirstPiolaStress(F, &P);

  Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
  model.CalcFirstPiolaStressDerivative(F, &dPdF);
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
