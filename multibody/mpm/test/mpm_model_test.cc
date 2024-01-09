#include "drake/multibody/mpm/mpm_model.h"

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {
using drake::multibody::mpm::constitutive_model::CorotatedElasticModel;
constexpr double kTolerance = 1e-10;
constexpr int num_active_nodes = 63;
// for the three particular particles we add, there will be a total of 63 active
// nodes.

Eigen::Matrix3X<AutoDiffXd> MakeMatrix3XWithDerivatives() {
  constexpr int columns = num_active_nodes;
  Eigen::Matrix3Xd X = Eigen::MatrixXd::Random(3, columns);
  const Eigen::Matrix<double, 3 * columns, Eigen::Dynamic> derivatives(
      Eigen::Matrix<double, 3 * columns, 3 * columns>::Identity());
  const auto X_autodiff_flat = math::InitializeAutoDiff(
      Eigen::Map<const Eigen::Matrix<double, 3 * columns, 1>>(X.data(),
                                                              3 * columns),
      derivatives);
  Eigen::Matrix3X<AutoDiffXd> X_autodiff =
      Eigen::Map<const Matrix3X<AutoDiffXd>>(X_autodiff_flat.data(), 3,
                                             columns);
  return X_autodiff;
}

GTEST_TEST(MpmModelTest, TestEnergyAndItsDerivatives) {
  // setup some particles and a grid.
  const double h = 0.2;
  double dt = 0.1;
  Particles<AutoDiffXd> particles = Particles<AutoDiffXd>();
  particles.AddParticle(
      Vector3<AutoDiffXd>(0.01, 0.01, 0.01), Vector3<AutoDiffXd>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<AutoDiffXd>>(1.0, 0.2), 1.0, 1.0);

  particles.AddParticle(
      Vector3<AutoDiffXd>(0.05, -0.05, 0.15),
      Vector3<AutoDiffXd>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<AutoDiffXd>>(3.0, 0.2), 1.0, 1.0);

  particles.AddParticle(
      Vector3<AutoDiffXd>(-1.2, 0.0, 0.4), Vector3<AutoDiffXd>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<AutoDiffXd>>(5.0, 0.2), 1.0, 1.0);

  SparseGrid<AutoDiffXd> sparse_grid(h);
  GridData<AutoDiffXd> grid_data{};

  MpmTransfer<AutoDiffXd> mpm_transfer{};
  TransferScratch<AutoDiffXd> transfer_scratch{};
  DeformationState<AutoDiffXd> deformation_state(particles, sparse_grid,
                                                 grid_data);

  // setup mpm_model and auxiliary scratch
  MpmModel<AutoDiffXd> mpm_model{};

  // given current particles (say just advect positions)
  mpm_transfer.SetUpTransfer(&sparse_grid, &particles);
  mpm_transfer.P2G(particles, sparse_grid, &grid_data, &transfer_scratch);
  EXPECT_EQ(grid_data.num_active_nodes(), num_active_nodes);
  // now we have G, can enter while loop

  // say we figured out a dG, let's get the new G

  // first save the current grid_v as v0
  std::vector<Vector3<AutoDiffXd>> v_prev = grid_data.velocities();

  // Manually fill in grid_data with random grid velocities, this is G += dG
  Eigen::Matrix3X<AutoDiffXd> Vi = MakeMatrix3XWithDerivatives();
  std::vector<Vector3<AutoDiffXd>> grid_velocities_input{};
  for (int i = 0; i < num_active_nodes; ++i) {
    grid_velocities_input.push_back(Vi.col(i));
  }
  grid_data.SetVelocities(grid_velocities_input);

  deformation_state.Update(mpm_transfer, dt, &transfer_scratch);
  // now we are ready to compute energy and its derivatives

  // first test elastic energy and its derivatives
  // they are the main components of total energy
  // @note elastic energy does not depend on v_prev

  // check that ComputeMinusDElasticEnergyDV() = - d (elastic_energy) / dv
  AutoDiffXd elastic_energy = mpm_model.ComputeElasticEnergy(deformation_state);
  Eigen::VectorX<double> d_elastic_energy_dv = elastic_energy.derivatives();
  Eigen::VectorX<AutoDiffXd> minus_d_elastic_energy_dv;
  mpm_model.ComputeMinusDElasticEnergyDV(mpm_transfer, deformation_state, dt,
                                         &minus_d_elastic_energy_dv,
                                         &transfer_scratch);
  EXPECT_TRUE(CompareMatrices(-d_elastic_energy_dv, minus_d_elastic_energy_dv,
                              kTolerance));

  // check that ComputeD2ElasticEnergyDV2() is correct
  MatrixX<AutoDiffXd> d2_elastic_energy_dv2;
  mpm_model.ComputeD2ElasticEnergyDV2(mpm_transfer, deformation_state, dt,
                                      &d2_elastic_energy_dv2);
  for (int i = 0; i < num_active_nodes; ++i) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      Eigen::VectorX<AutoDiffXd> hessian_slice =
          d2_elastic_energy_dv2.col(i * 3 + alpha);
      // .col or .row should be the same, since it is symmetric
      Eigen::VectorX<double> d_minus_d_elastic_energy_dv_i_alpha_dv =
          minus_d_elastic_energy_dv(3 * i + alpha).derivatives();

      EXPECT_TRUE(CompareMatrices(
          hessian_slice, -d_minus_d_elastic_energy_dv_i_alpha_dv, kTolerance));
    }
  }

  // next we test the total energy, which involves v_prev, save flow as above
  // energy
  AutoDiffXd energy = mpm_model.ComputeEnergy(v_prev, deformation_state, dt);
  // -dedv
  Eigen::VectorX<AutoDiffXd> minus_d_energy_dv;
  mpm_model.ComputeMinusDEnergyDV(mpm_transfer, v_prev, deformation_state, dt,
                                  &minus_d_energy_dv, &transfer_scratch);

  Eigen::VectorX<double> d_energy_dv = energy.derivatives();
  EXPECT_TRUE(CompareMatrices(-d_energy_dv, minus_d_energy_dv, kTolerance));

  MatrixX<AutoDiffXd> d2edv2;
  mpm_model.ComputeD2EnergyDV2(mpm_transfer, deformation_state, dt, &d2edv2);
  for (int i = 0; i < num_active_nodes; ++i) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      Eigen::VectorX<AutoDiffXd> hessian_slice = d2edv2.col(i * 3 + alpha);
      // .col or .row should be the same, since it is symmetric
      Eigen::VectorX<double> d_minus_d_energy_dv_i_alpha_dv =
          minus_d_energy_dv(3 * i + alpha).derivatives();

      EXPECT_TRUE(CompareMatrices(hessian_slice,
                                  -d_minus_d_energy_dv_i_alpha_dv, kTolerance));
    }
  }
}

GTEST_TEST(MpmModelTest, TestHessianTimesZ) {
  // now we already know hessian is correct, we just test HessianTimesZ(z) =
  // hessian * z, where hessian is either the elastic hessian or the total
  // hessian

  const double h = 0.2;
  double dt = 0.1;
  Particles<double> particles = Particles<double>();
  particles.AddParticle(
      Vector3<double>(0.01, 0.01, 0.01), Vector3<double>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<double>>(1.0, 0.2), 1.0, 1.0);

  particles.AddParticle(
      Vector3<double>(0.05, -0.05, 0.15), Vector3<double>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<double>>(3.0, 0.2), 1.0, 1.0);

  particles.AddParticle(
      Vector3<double>(-1.2, 0.0, 0.4), Vector3<double>(0.0, 0.0, 0.0),
      std::make_unique<CorotatedElasticModel<double>>(5.0, 0.2), 1.0, 1.0);

  SparseGrid<double> sparse_grid(h);
  GridData<double> grid_data{};

  MpmTransfer<double> mpm_transfer{};
  TransferScratch<double> transfer_scratch{};
  DeformationState<double> deformation_state(particles, sparse_grid, grid_data);

  // setup mpm_model and auxiliary scratch
  MpmModel<double> mpm_model{};

  mpm_transfer.SetUpTransfer(&sparse_grid, &particles);
  mpm_transfer.P2G(particles, sparse_grid, &grid_data, &transfer_scratch);

  // now randomly modify some grid velocities
  Eigen::Matrix3Xd V =
      Eigen::MatrixXd::Random(3, sparse_grid.num_active_nodes());
  std::vector<Vector3<double>> grid_velocities_input{};
  for (size_t i = 0; i < sparse_grid.num_active_nodes(); ++i) {
    grid_velocities_input.push_back(V.col(i));
  }
  grid_data.SetVelocities(grid_velocities_input);
  deformation_state.Update(mpm_transfer, dt, &transfer_scratch);

  Eigen::VectorXd z(3 * sparse_grid.num_active_nodes());
  Eigen::VectorXd hessian_times_z(3 * sparse_grid.num_active_nodes());

  // elastic hessian
  MatrixX<double> elastic_hessian;
  mpm_model.ComputeD2ElasticEnergyDV2(mpm_transfer, deformation_state, dt,
                                      &elastic_hessian);

  // test a few random vectors
  for (int i = 0; i < 20; ++i) {
    z.setRandom();
    hessian_times_z.setZero();
    mpm_model.AddD2ElasticEnergyDV2TimesZ(z, mpm_transfer, deformation_state,
                                          dt, &hessian_times_z);

    EXPECT_TRUE(
        CompareMatrices(elastic_hessian * z, hessian_times_z, kTolerance));
  }

  // total hessian
  MatrixX<double> hessian;
  mpm_model.ComputeD2EnergyDV2(mpm_transfer, deformation_state, dt, &hessian);

  // test a few random vectors
  for (int i = 0; i < 20; ++i) {
    z.setRandom();
    hessian_times_z.setZero();
    mpm_model.AddD2EnergyDV2TimesZ(z, mpm_transfer, deformation_state, dt,
                                   &hessian_times_z);

    EXPECT_TRUE(CompareMatrices(hessian * z, hessian_times_z, kTolerance));
  }
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
