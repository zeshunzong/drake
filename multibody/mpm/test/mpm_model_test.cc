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

template <typename T>
void VecXToStdvecVec3(const Eigen::VectorX<T> input,
                      std::vector<Vector3<T>>* output) {
  size_t length = input.size() / 3;
  std::vector<Vector3<T>>& output_ref = *output;
  output->resize(length);
  for (size_t i = 0; i < length; ++i) {
    output_ref[i] =
        Vector3<T>(input(3 * i), input(3 * i + 1), input(3 * i + 2));
  }
}

template <typename T>
void StdvecVec3ToVecX(const std::vector<Vector3<T>>& input,
                      Eigen::VectorX<T>* output) {
  Eigen::VectorX<T>& output_ref = *output;
  output_ref.resize(input.size() * 3);
  for (size_t i = 0; i < input.size(); ++i) {
    output_ref(3 * i) = input[i](0);
    output_ref(3 * i + 1) = input[i](1);
    output_ref(3 * i + 2) = input[i](2);
  }
}

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

GTEST_TEST(MpmModelTest, TestEnergyAndForceAndHessian) {
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
  // Manually fill in grid_data with random grid velocities
  Eigen::Matrix3X<AutoDiffXd> Vi = MakeMatrix3XWithDerivatives();
  std::vector<Vector3<AutoDiffXd>> grid_velocities_input{};
  for (int i = 0; i < num_active_nodes; ++i) {
    grid_velocities_input.push_back(Vi.col(i));
  }
  grid_data.SetVelocities(grid_velocities_input);

  // now we can compute the energy and force for the next while loop
  deformation_state.Update(mpm_transfer, dt, &transfer_scratch);

  // check that elastic force is derivative of elastic energy (d energy / dx)

  // energy
  AutoDiffXd energy = mpm_model.ComputeElasticEnergy(deformation_state);
  // force
  std::vector<Vector3<AutoDiffXd>> elastic_forces;
  mpm_model.ComputeElasticForce(mpm_transfer, deformation_state,
                                &elastic_forces, &transfer_scratch);

  Eigen::VectorX<AutoDiffXd> dEnergydV = energy.derivatives();
  // this should be of size num_active_nodes * 3
  Eigen::VectorX<AutoDiffXd> dEnergydX = dEnergydV / dt;  // chain rule

  // ---------------- check force = -dedx ------------
  for (int i = 0; i < num_active_nodes; ++i) {
    Eigen::VectorX<AutoDiffXd> dEnergydXi = dEnergydX.segment(3 * i, 3);
    EXPECT_TRUE(CompareMatrices(-dEnergydXi, elastic_forces[i], kTolerance));
  }

  // ---------------- check hessian = d^2 e d x^2 ------------
  MatrixX<AutoDiffXd> hessian;
  mpm_model.ComputeElasticHessian(mpm_transfer, deformation_state, &hessian);
  for (int i = 0; i < num_active_nodes; ++i) {
    for (int alpha = 0; alpha < 1; ++alpha) {
      Eigen::VectorX<AutoDiffXd> hessian_slice = hessian.col(i * 3 + alpha);
      // .col or .row should be the same, since it is symmetric
      Eigen::VectorX<AutoDiffXd> grid_force_i = elastic_forces[i];
      MatrixX<AutoDiffXd> d_grid_force_ialpha_d_V =
          grid_force_i(alpha).derivatives();  // 1 by 3*num_active_grids
      MatrixX<AutoDiffXd> d_grid_force_ialpha_d_X =
          d_grid_force_ialpha_d_V / dt;  // chain rule
      EXPECT_TRUE(
          CompareMatrices(hessian_slice, -d_grid_force_ialpha_d_X, kTolerance));
    }
  }
}

GTEST_TEST(MpmModelTest, TestHessianTimesZ) {
  // now we already know hessian is correct, we just test HessianTimesZ(z) =
  // hessian * z;

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

  MatrixX<double> hessian;
  mpm_model.ComputeElasticHessian(mpm_transfer, deformation_state, &hessian);

  Eigen::VectorX<double> z_vec;
  z_vec.resize(3 * sparse_grid.num_active_nodes());
  std::vector<Vector3<double>> z_stdvec;
  std::vector<Vector3<double>> hessian_times_z_computed;
  Eigen::VectorX<double> hessian_times_z_computed_vec;

  // test a few random vectors
  for (int i = 0; i < 20; ++i) {
    z_vec.setRandom();
    VecXToStdvecVec3<double>(z_vec, &z_stdvec);
    mpm_model.ComputeElasticHessianTimesZ(
        z_stdvec, mpm_transfer, deformation_state, &hessian_times_z_computed);

    StdvecVec3ToVecX<double>(hessian_times_z_computed,
                             &hessian_times_z_computed_vec);
    EXPECT_TRUE(CompareMatrices(hessian * z_vec, hessian_times_z_computed_vec,
                                kTolerance));
  }
}
}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
