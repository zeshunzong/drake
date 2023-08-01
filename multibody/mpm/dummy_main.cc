#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/SparseGrid.h"
#include "drake/multibody/mpm/MPMTransfer.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include <iostream>
#include <vector>
#include "drake/multibody/plant/deformable_model.h"
#include <stdlib.h> 

namespace drake {
namespace multibody {
namespace mpm {


using Eigen::Matrix3d;
using Eigen::Matrix3Xd;

Eigen::Matrix3X<AutoDiffXd> MakeMatrix3XWithDerivatives(){
    constexpr int columns=42;
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

int DoMain() {

    mpm::Particles<AutoDiffXd> particles(0);
    SparseGrid<AutoDiffXd> grid{0.2};
    MPMTransfer<AutoDiffXd> mpm_transfer{};
    CorotatedElasticModel<AutoDiffXd> model(2.5,0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p = model.Clone();

    Eigen::Matrix3<AutoDiffXd> F_in;
    F_in.setIdentity();
    F_in(0,2) = 0.2;
    F_in(2,1) = 0.3;

    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.1,0.0,0.0}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 5.0, 1.0,
            F_in, Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p));

    Eigen::Matrix3<AutoDiffXd> F_in2;
    F_in2.setIdentity();
    F_in2(0,2) = 0.2;
    F_in2(2,1) = -0.1;
    F_in2(0,1) = 0.3;
    std::unique_ptr<mpm::ElastoPlasticModel<AutoDiffXd>> elastoplastic_model_p2 = model.Clone();
    particles.AddParticle(Eigen::Vector3<AutoDiffXd>{0.3,0.2,0.15}, Eigen::Vector3<AutoDiffXd>{0,0,0}, 3.0, 1.0,
            F_in2, Eigen::Matrix3<AutoDiffXd>::Identity(), 
            Eigen::Matrix3<AutoDiffXd>::Identity(),Eigen::Matrix3<AutoDiffXd>::Zero(), std::move(elastoplastic_model_p2));
 
    
    int num_active_grids = mpm_transfer.MakeGridCompatibleWithParticles(&particles, &grid);
    std::cout << num_active_grids << std::endl; getchar();
    // // EXPECT_EQ(num_active_grids, 42);

    // Temporary----Manually set up vi*
    Eigen::Matrix3X<AutoDiffXd> Vi = MakeMatrix3XWithDerivatives();
    std::vector<Eigen::Vector3<AutoDiffXd>> grid_velocities_input{};
    for (size_t i = 0; i < 42; i++){
        grid_velocities_input.push_back(Vi.col(i));
    }
    mpm_transfer.DerivativeTest2(&grid, grid_velocities_input);
    // Temporary----Manually set up vi*

    AutoDiffXd dt = 0.1;
    MatrixX<AutoDiffXd> hessian;
    AutoDiffXd energy = mpm_transfer.computeEnergyForceHessian(&particles, &grid, &hessian, dt);
    Eigen::VectorX<AutoDiffXd> ddd = energy.derivatives();


    std::cout << ddd(0) << " " << ddd(1) << " " << ddd(2) << " " << ddd(3) << std::endl;
    std::cout << grid.get_force(0)[0] * dt << " " << grid.get_force(0)[1] * dt << " " << grid.get_force(0)[2] * dt << " " << grid.get_force(1)[0] * dt <<std::endl;

    Eigen::VectorX<AutoDiffXd> gf= grid.get_force(0);
    MatrixX<AutoDiffXd> xxx = gf(0).derivatives();
    for (int i = 0; i < 20; ++i) {
        std::cout << xxx(i)<< " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << hessian(0,i) * dt << " ";
    }

    Eigen::VectorX<AutoDiffXd> hessian_slice = hessian.col(0);   
    Eigen::VectorX<AutoDiffXd> diff = (hessian_slice * dt + xxx);

    std::cout << std::endl;
    std::cout << diff.sum() << std::endl;
    



    for (int i = 0; i < 42; i++) {
        for (int j = 0; j < 3; j++) {
            Eigen::VectorX<AutoDiffXd> gf2 = grid.get_force(i);
            MatrixX<AutoDiffXd> xxx2 = gf2(j).derivatives();
            Eigen::VectorX<AutoDiffXd> hessian_slice2 = hessian.col(i*3+j);
            Eigen::VectorX<AutoDiffXd> diff2 = (hessian_slice2 * dt + xxx2);
            std::cout << "dif " << diff2.sum() << std::endl;
        }
    }

    //Eigen::Map<Eigen::VectorXd> flattened(matrix.data(), matrix.size());
    // std::cout << hessian(3,3) * dt << std::endl;
    // Eigen::VectorX<AutoDiffXd> gf1= grid.get_force(1);
    // MatrixX<AutoDiffXd> xxx1 = gf1(0).derivatives();
    // std::cout << xxx1(3) << std::endl;
    return 0;


}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main() {
    return drake::multibody::mpm::DoMain();
}