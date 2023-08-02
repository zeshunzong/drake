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
#include "drake/common/eigen_types.h"


namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

// constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;


Eigen::Matrix3X<AutoDiffXd> MakeMatrix3XWithDerivatives(){
    constexpr int columns=52;
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


void TestEnergyAndForceAndHessian(){

    // initialize grid and transfer
    mpm::Particles<AutoDiffXd> particles(0);
    SparseGrid<AutoDiffXd> grid{0.2};
    MPMTransfer<AutoDiffXd> mpm_transfer{};


    // add particle #1
    Eigen::Matrix3<AutoDiffXd> F_in;
    F_in.setIdentity(); F_in(0,2) = 0.2; F_in(2,1) = 0.3;
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p = model.Clone();
    // particle volume = 5;
    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.1,-0.37,0.22}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 5.0, 1.0,
            F_in, Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p));

    // add particle #2
    Eigen::Matrix3<AutoDiffXd> F_in2;
    F_in2.setIdentity(); F_in2(0,1) = -0.17; F_in2(2,1) = 0.18; F_in2(2,2) = 0.9;
    CorotatedElasticModel<AutoDiffXd> model2(2.7,0.24);
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p2 = model2.Clone();
    // particle volume = 3;
    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.07,-0.07,-0.12}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 3.0, 1.0,
            F_in2, Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p2));

    // add particle #3
    Eigen::Matrix3<AutoDiffXd> F_in3;
    F_in3.setIdentity(); F_in3(1,0) = 0.15; F_in3(1,1) = 1.18;
    CorotatedElasticModel<AutoDiffXd> model3(12.1,0.24);
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p3 = model3.Clone();
    // particle volume = 4.1;
    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.2,-0.4,0.25}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 4.1, 1.0,
            F_in3, Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p3));

    int num_active_grids = mpm_transfer.MakeGridCompatibleWithParticles(&particles, &grid);

    std::cout << "active : " << num_active_grids << std::endl;

    EXPECT_EQ(num_active_grids, 52);
    // hardcoded 
    // Temporary----Manually set up vi*
    Eigen::Matrix3X<AutoDiffXd> Vi = MakeMatrix3XWithDerivatives();
    std::vector<Eigen::Vector3<AutoDiffXd>> grid_velocities_input{};
    for (int i = 0; i < num_active_grids; i++){
        grid_velocities_input.push_back(Vi.col(i));
    }
    grid.OverwriteGridVelocity(grid_velocities_input); // feed in indep autodiff vars in grid v
    // Temporary----Manually set up vi*

    AutoDiffXd dt = 0.1;
    MatrixX<AutoDiffXd> hessian;
    AutoDiffXd energy = mpm_transfer.CalcEnergyForceHessian(&particles, &grid, &hessian, dt);
    Eigen::VectorX<AutoDiffXd> dEnergydV = energy.derivatives(); // this should be num_active_nodes * 3

    Eigen::VectorX<AutoDiffXd> dEnergydX = dEnergydV/dt; // chain rule

    // check force = -dedx
    for (int i = 0; i < num_active_grids; i++) {
        Eigen::VectorX<AutoDiffXd> dEnergydXi = dEnergydX.segment(3*i, 3);
        Eigen::VectorX<AutoDiffXd> force_i = grid.get_force(i);

        EXPECT_TRUE(CompareMatrices(-dEnergydXi, force_i, 1e-10));
    }

    // check hessian = -dfdx
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            Eigen::VectorX<AutoDiffXd> grid_force_i = grid.get_force(i);
            MatrixX<AutoDiffXd> d_grid_force_ij_d_V = grid_force_i(j).derivatives(); // 1 by 3*num_active_grids
            MatrixX<AutoDiffXd> d_grid_force_ij_d_X = d_grid_force_ij_d_V/dt; // chain rule
            Eigen::VectorX<AutoDiffXd> hessian_slice = hessian.col(i*3+j); //.col or .row should be the same, since it is symmetric
            EXPECT_TRUE(CompareMatrices(-d_grid_force_ij_d_X, hessian_slice, 1e-10));
        }
    }

    // check hessian is symmetric
    EXPECT_TRUE(CompareMatrices(hessian, hessian.transpose(), 1e-10));


}


GTEST_TEST(TestEnergyForceHessian, energyforcehessian) {
    TestEnergyAndForceAndHessian();
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
