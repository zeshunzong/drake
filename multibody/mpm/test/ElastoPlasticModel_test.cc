#include "drake/multibody/fem/mpm-dev/ElastoPlasticModel.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/fem/mpm-dev/CorotatedElasticModel.h"
#include "drake/multibody/fem/mpm-dev/StvkHenckyWithVonMisesModel.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double TOLERANCE = 1e-10;

GTEST_TEST(CorotatedModelTest, CorotatedModelAllTest) {
    // Taken from fem/test/corotated_model_data_test.cc
    /* We set deformation gradient as F = R*S where R is an arbitrary rotation
    matrix and S is an arbitrary sysmmetric positive definite matrix. */
    const Matrix3<double> R =
        math::RotationMatrix<double>
            (math::RollPitchYaw<double>(pi/2.0, pi, 3.0*pi/2.0)).matrix();
    const Matrix3<double> S = (Matrix3<double>() <<
            6, 1, 2,
            1, 4, 1,
            2, 1, 5).finished();
    const Matrix3<double> tau_exact1 = (Matrix3<double>() <<
             50, -42,  20,
            -42,  70, -22,
             20, -22,  28).finished();
    const Matrix3<double> tau_exact2 = (Matrix3<double>() <<
             18724, -84,  40,
            -84,  18764, -44,
             40, -44,  18680).finished();
    Matrix3<double> F = R * S;
    Matrix3<double> F2;
    Matrix3<double> tau;

    // First test on E = 2.0, nu = 0.0
    CorotatedElasticModel corotated_model = CorotatedElasticModel(2.0, 0.0);
    F2 = F;
    corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    EXPECT_TRUE(CompareMatrices(tau, tau_exact1, TOLERANCE));
    // Sanity check: no plasticity shall be applied
    EXPECT_TRUE(CompareMatrices(F, F2, TOLERANCE));

    // Next test on E = 5.0, nu = 0.25
    corotated_model = CorotatedElasticModel(5.0, 0.25);
    F2 = F;
    corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F);
    EXPECT_TRUE(CompareMatrices(tau, tau_exact2, TOLERANCE));
    // Sanity check: no plasticity shall be applied
    EXPECT_TRUE(CompareMatrices(F, F2, TOLERANCE));

    // Sanity check: If F is a rotation matrix, then stresses are zero
    corotated_model = CorotatedElasticModel(5.0, 0.25);
    F2 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(1.0, 2.0, 3.0)).matrix();
    corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F2);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), TOLERANCE));

    corotated_model = CorotatedElasticModel(23.4, 0.41);
    F2 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(0.1, -2.4, 13.3)).matrix();
    corotated_model.UpdateDeformationGradientAndCalcKirchhoffStress(&tau, &F2);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), TOLERANCE));
}

GTEST_TEST(StvkHenckyWithVonMisesModelTest,
           StvkHenckyWithVonMisesModelConstitutiveModelTest) {
    // Set yield stress large enough so no plasticity is applied
    Matrix3<double> tau, dummy_FE;
    double E0      = 5.0;
    double nu0     = 0.25;
    double tau_c0  = 1e6;
    double E1      = 23.4;
    double nu1     = 0.41;
    double tau_c1  = 2e6;
    StvkHenckyWithVonMisesModel hencky_model0 = {E0, nu0, tau_c0};
    StvkHenckyWithVonMisesModel hencky_model1 = {E1, nu1, tau_c1};

    // Sanity check: If F is a rotation matrix, then stresses are zero
    Matrix3<double> F0 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(1.0, 2.0, 3.0)).matrix();
    dummy_FE = F0;
    hencky_model0.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
                                                                  &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), TOLERANCE));

    Matrix3<double> F1 = math::RotationMatrix<double>
                (math::RollPitchYaw<double>(0.1, -2.4, 13.3)).matrix();
    dummy_FE = F1;
    hencky_model1.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
                                                                  &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, Matrix3<double>::Zero(), TOLERANCE));

    // For another F with SVD
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
    Matrix3<double> F2 =
    (Matrix3<double>() << 3/sqrt(2), 3/sqrt(2),        0.0,
                                 -1,         1, -1/sqrt(2),
                                 -1,         1,  1/sqrt(2)).finished();
    Matrix3<double> U =
    (Matrix3<double>() << 1.0,       0.0,        0.0,
                          0.0, sqrt(2)/2, -sqrt(2)/2,
                          0.0, sqrt(2)/2,  sqrt(2)/2).finished();
    Matrix3<double> V =
    (Matrix3<double>() << sqrt(2)/2, -sqrt(2)/2, 0.0,
                          sqrt(2)/2,  sqrt(2)/2, 0.0,
                                0.0,        0.0, 1.0).finished();
    double sum_log_sigma = std::log(3)+std::log(2);

    // Check with the first hencky model
    double mu0     = E0/(2*(1+nu0));
    double lambda0 = E0*nu0/(1+nu0)/(1-2*nu0);
    Vector3<double> sigma0 = {2.0/3.0*mu0*std::log(3)+lambda0/3.0*sum_log_sigma,
                                      mu0*std::log(2)+lambda0/2.0*sum_log_sigma,
                                                      lambda0*sum_log_sigma};
    Matrix3<double> P0   = U*sigma0.asDiagonal()*V.transpose();
    Matrix3<double> tau0 = P0*F2.transpose();
    dummy_FE = F2;
    hencky_model0.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
                                                                  &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, tau0, TOLERANCE));

    // Check with the next hencky model
    double mu1     = E1/(2*(1+nu1));
    double lambda1 = E1*nu1/(1+nu1)/(1-2*nu1);
    Vector3<double> sigma1 = {2.0/3.0*mu1*std::log(3)+lambda1/3.0*sum_log_sigma,
                                      mu1*std::log(2)+lambda1/2.0*sum_log_sigma,
                                                      lambda1*sum_log_sigma};
    Matrix3<double> P1   = U*sigma1.asDiagonal()*V.transpose();
    Matrix3<double> tau1 = P1*F2.transpose();
    dummy_FE = F2;
    hencky_model1.UpdateDeformationGradientAndCalcKirchhoffStress(&tau,
                                                                  &dummy_FE);
    EXPECT_TRUE(CompareMatrices(tau, tau1, TOLERANCE));
}

GTEST_TEST(StvkHenckyWithVonMisesModelTest,
           StvkHenckyWithVonMisesModelPlasticModelTest) {
    Matrix3<double> FE = (Matrix3<double>() <<
            6, 1, 2,
            1, 4, 1,
            2, 1, 5).finished();

    double E      = 100.0;
    double nu     = 0.2;
    // We set a tau_c large enough so that the current elastic deformation
    // gradient is in the yield surface
    double tau_c0  = 1000.0;
    std::unique_ptr<StvkHenckyWithVonMisesModel> hencky_model0
                = std::make_unique<StvkHenckyWithVonMisesModel>(E, nu, tau_c0);
    // Deformation gradient before plasticity
    Matrix3<double> FEprev = FE;
    // The trial stress is in the yield surface if
    // sqrt(3/2)‖ dev(τ) ‖_F ≤ τ_c
    bool in_yield_surface0 = (hencky_model0->EvalYieldFunction(FE)
                           <= TOLERANCE);
    EXPECT_TRUE(in_yield_surface0);

    // Apply plasticity and calculate the new Kirchhoff Stress
    Matrix3<double> tau0;
    hencky_model0->UpdateDeformationGradientAndCalcKirchhoffStress(&tau0, &FE);
    in_yield_surface0 = (hencky_model0->EvalYieldFunction(FE) <= TOLERANCE);

    // Nothing shall change in this case
    EXPECT_TRUE(in_yield_surface0);
    EXPECT_TRUE(CompareMatrices(FEprev, FE, TOLERANCE));

    // We set a tau_c small enough so that the current elastic deformation
    // gradient is not in the yield surface
    double tau_c1  = 70.0;
    std::unique_ptr<StvkHenckyWithVonMisesModel> hencky_model1
                = std::make_unique<StvkHenckyWithVonMisesModel>(E, nu, tau_c1);
    // Deformation gradient before plasticity
    FEprev = FE;
    bool in_yield_surface1 = (hencky_model1->EvalYieldFunction(FE)
                           <= TOLERANCE);
    EXPECT_FALSE(in_yield_surface1);

    // Apply plasticity and calculate the new Kirchhoff Stress
    Matrix3<double> tau1;
    hencky_model1->UpdateDeformationGradientAndCalcKirchhoffStress(&tau1, &FE);
    bool on_yield_surface1 = (abs(hencky_model1->EvalYieldFunction(FE))
                           <= TOLERANCE);

    EXPECT_TRUE(on_yield_surface1);
    EXPECT_FALSE(CompareMatrices(FEprev, FE, TOLERANCE));
}


}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

