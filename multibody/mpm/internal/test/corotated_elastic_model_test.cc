
#include <gtest/gtest.h>

#include "drake/multibody/mpm/internal/elastoplastic_model.h"
#include "drake/multibody/mpm/internal/corotated_elastic_model.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"


namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double kTolerance = 1e-10;

GTEST_TEST(CorotatedModelTest, CorotatedModelAllTest) {
    // Taken from fem/test/corotated_model_data_test.cc
    /* We set deformation gradient as F = R*S where R is an arbitrary rotation
    matrix and S is an arbitrary sysmmetric positive definite matrix. */
    // const Matrix3<double> R =
    //     math::RotationMatrix<double>
    //         (math::RollPitchYaw<double>(pi/2.0, pi, 3.0*pi/2.0)).matrix();
    // const Matrix3<double> S = (Matrix3<double>() <<
    //         6, 1, 2,
    //         1, 4, 1,
    //         2, 1, 5).finished();
    // const Matrix3<double> tau_exact1 = (Matrix3<double>() <<
    //          50, -42,  20,
    //         -42,  70, -22,
    //          20, -22,  28).finished();
    // // const Matrix3<double> tau_exact2 = (Matrix3<double>() <<
    // //          18724, -84,  40,
    // //         -84,  18764, -44,
    // //          40, -44,  18680).finished();
    // Matrix3<double> F = R * S;
    // Matrix3<double> F2;
    // Matrix3<double> tau;

    // // First test on E = 2.0, nu = 0.0
    CorotatedElasticModel<double> model(2.5,0.3);
    //CorotatedElasticModel<double> corotated_model = CorotatedElasticModel<double>();//(2.0, 0.0);
    // F2 = F;
    // corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    // EXPECT_TRUE(CompareMatrices(tau, tau_exact1, kTolerance));
    // // Sanity check: no plasticity shall be applied
    // EXPECT_TRUE(CompareMatrices(F, F2, kTolerance));

    // // Next test on E = 5.0, nu = 0.25
    // corotated_model = CorotatedElasticModel(5.0, 0.25);
    // F2 = F;
    // corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    // EXPECT_TRUE(CompareMatrices(tau, tau_exact2, kTolerance));
    // // Sanity check: no plasticity shall be applied
    // EXPECT_TRUE(CompareMatrices(F, F2, kTolerance));

    // // Sanity check: If F is a rotation matrix, then stresses are zero
    // corotated_model = CorotatedElasticModel(5.0, 0.25);
    // F2 = math::RotationMatrix<double>
    //             (math::RollPitchYaw<double>(1.0, 2.0, 3.0)).matrix();
    // corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F2);
    // EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), kTolerance));

    // corotated_model = CorotatedElasticModel(23.4, 0.41);
    // F2 = math::RotationMatrix<double>
    //             (math::RollPitchYaw<double>(0.1, -2.4, 13.3)).matrix();
    // corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F2);
    // EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), kTolerance));
}


}  // namespace
}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

