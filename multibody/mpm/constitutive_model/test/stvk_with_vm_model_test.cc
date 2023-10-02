#include <stdlib.h>

#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/mpm/constitutive_model/StvkHenckyWithVonMisesModel.h"
#include "drake/multibody/mpm/constitutive_model/StvkHenckyWithVonMisesModel2.h"


namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {
namespace {

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;

constexpr double pi = 3.14159265358979323846;
constexpr double kTolerance = 1e-10;

// Matrix3<AutoDiffXd> MakeDeformationGradientsWithDerivatives() {
//   /* Create an arbitrary AutoDiffXd deformation. */
//   Matrix3d F;
//   // clang-format off
//   F << 0.18, 0.63, 0.54,
//        0.13, 0.92, 0.17,
//        0.03, 0.86, 0.85;
//   // clang-format on
//   const Eigen::Matrix<double, 9, Eigen::Dynamic> derivatives(
//       Eigen::Matrix<double, 9, 9>::Identity());
//   const auto F_autodiff_flat = math::InitializeAutoDiff(
//       Eigen::Map<const Eigen::Matrix<double, 9, 1>>(F.data(), 9), derivatives);
//   Matrix3<AutoDiffXd> deformation_gradient_autodiff =
//       Eigen::Map<const Matrix3<AutoDiffXd>>(F_autodiff_flat.data(), 3, 3);
//   return deformation_gradient_autodiff;
// }

GTEST_TEST(CorotatedElasticModelTest, TestReturnMapAndStress) {
    double E0      = 5.0;
    double nu0     = 0.25;
    double tau_c0  = 1e6;
    double E1      = 23.4;
    double nu1     = 0.41;
    double tau_c1  = 2e6;
    StvkHenckyWithVonMisesModel<double> hencky_model0(E0, nu0, tau_c0);
    StvkHenckyWithVonMisesModel2<double> hencky_model1(E1, nu1, tau_c1);
    Matrix3<double> tau, dummy_FE;
    // Sanity check: If F is a rotation matrix, then stresses are zero
    Matrix3<double> F0 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(1.0, 2.0, 3.0)).matrix();
    dummy_FE = F0;
    hencky_model0.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
                                                                  &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), kTolerance));

    Matrix3<double> F1 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(0.1, -2.4, 13.3)).matrix();
    dummy_FE = F1;
    hencky_model1.CalcFEFromFtrial(&dummy_FE);
    hencky_model1.CalcKirchhoffStress(dummy_FE, &tau);
    // hencky_model1.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
    //                                                               &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), kTolerance));

}

// GTEST_TEST(CorotatedElasticModelTest, TestEnergyDerivatives) {
//   // Test P is derivative of Psi
//   CorotatedElasticModel<AutoDiffXd> model(3.14159, 0.3337);
//   Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();
//   AutoDiffXd f = model.CalcStrainEnergyDensity(F);
//   Matrix3<AutoDiffXd> dfdF;
//   model.CalcFirstPiolaStress(F, &dfdF);
//   EXPECT_TRUE(
//       CompareMatrices(Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3),
//                       dfdF, kTolerance));

//   // Test dPdF is derivative of P
//   Matrix3<AutoDiffXd> P;
//   model.CalcFirstPiolaStress(F, &P);

//   Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
//   model.CalcFirstPiolaStressDerivative(F, &dPdF);
//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       Matrix3d dPijdF;
//       for (int k = 0; k < 3; ++k) {
//         for (int l = 0; l < 3; ++l) {
//           dPijdF(k, l) = dPdF(3 * j + i, 3 * l + k).value();
//         }
//       }
//       EXPECT_TRUE(CompareMatrices(
//           Eigen::Map<const Matrix3d>(P(i, j).derivatives().data(), 3, 3),
//           dPijdF, kTolerance));
//     }
//   }
// }

}  // namespace
}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
