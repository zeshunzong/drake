#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"

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

GTEST_TEST(LinearCorotatedModelTest, TestEnergyDerivatives) {
  // Test P is derivative of Psi
  LinearCorotatedModel<AutoDiffXd> model(3.14159, 0.3337);
  Matrix3<AutoDiffXd> F = MakeDeformationGradientsWithDerivatives();
  Matrix3<AutoDiffXd> F0;
  F0.setRandom();
  AutoDiffXd f = model.CalcStrainEnergyDensity(F0, F);
  Matrix3<AutoDiffXd> dfdF;
  model.CalcFirstPiolaStress(F0, F, &dfdF);
  EXPECT_TRUE(
      CompareMatrices(Eigen::Map<const Matrix3d>(f.derivatives().data(), 3, 3),
                      dfdF, kTolerance));

  // Test dPdF is derivative of P
  Matrix3<AutoDiffXd> P;
  model.CalcFirstPiolaStress(F0, F, &P);

  Eigen::Matrix<AutoDiffXd, 9, 9> dPdF;
  model.CalcFirstPiolaStressDerivative(F0, F, &dPdF);
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
