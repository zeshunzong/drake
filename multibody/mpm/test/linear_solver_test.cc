#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/mpm/conjugate_gradient.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/matrix_replacement.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {
using drake::multibody::mpm::constitutive_model::CorotatedElasticModel;
constexpr double kTolerance = 1e-5;

GTEST_TEST(LinearSolverTest, TestCG) {
  double relative_tolerance = 1e-5;
  // want |b-Ax|/|b| < relative_tolerance
  // that is, |b-Ax| < relative_tolerance * |b|
  // so |b-Ax|∞ ≤ |b-Ax|₂ < relative_tolerance * |b|₂
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
  // this is G
  deformation_state.Update(mpm_transfer, dt, &transfer_scratch);

  // from G we can get hessian.
  // need to solve for x such that hessian * x = b (where b will be force, here
  // we randomly pick b)
  Eigen::VectorXd b(3 * sparse_grid.num_active_nodes());
  b.setRandom();

  // --- method 1: solve with Eigen::CG + dense matrix + diagonal preconditioner
  Eigen::VectorXd x_dense;
  MatrixX<double> hessian_dense;
  mpm_model.ComputeD2EnergyDV2(mpm_transfer, deformation_state, dt,
                               &hessian_dense);

  Eigen::ConjugateGradient<MatrixX<double>, Eigen::Lower | Eigen::Upper>
      cg_dense;
  cg_dense.setTolerance(relative_tolerance);
  // set same tol for each version of cg
  cg_dense.compute(hessian_dense);
  x_dense = cg_dense.solve(b);
  std::cout << "#iterations:     " << cg_dense.iterations() << std::endl;
  std::cout << "estimated error: " << cg_dense.error() << std::endl;

  // check that the result is correct
  EXPECT_TRUE(CompareMatrices(b - hessian_dense * x_dense,
                              Eigen::VectorXd::Zero(b.size()),
                              relative_tolerance * b.norm()));
  // --- method 1: solve with Eigen::CG + dense matrix + diagonal preconditioner

  // --- method 2: solve with Eigen::CG + matrix free + NO preconditioner
  MatrixReplacement<double> hessian_matrix_free =
      MatrixReplacement<double>(mpm_model, deformation_state, mpm_transfer, dt);
  Eigen::VectorXd x_matrix_free;

  Eigen::ConjugateGradient<MatrixReplacement<double>,
                           Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
      cg_matrix_free;
  cg_matrix_free.setTolerance(relative_tolerance);
  // set same tol for each version of cg
  cg_matrix_free.compute(hessian_matrix_free);
  x_matrix_free = cg_matrix_free.solve(b);
  std::cout << "CG_matrix free:#iterations: " << cg_matrix_free.iterations()
            << "\n"
            << "estimated error: " << cg_matrix_free.error() << std::endl;

  // check that the result is correct
  EXPECT_TRUE(CompareMatrices(b - hessian_dense * x_matrix_free,
                              Eigen::VectorXd::Zero(b.size()),
                              relative_tolerance * b.norm()));
  // --- method 2: solve with Eigen::CG + matrix free + NO preconditioner
  // @note method 2 needs considerably more iterations due to no preconditioner

  // --- method 3: solve with custom CG + matrix free + mass preconditioner
  HessianWrapper hessian_wrapper(mpm_transfer, mpm_model, deformation_state,
                                 dt);

  ConjugateGradient cg;
  cg.SetRelativeTolerance(relative_tolerance);
  Eigen::VectorXd x;
  cg.Solve(hessian_wrapper, b, &x);
  EXPECT_TRUE(CompareMatrices(b - hessian_dense * x,
                              Eigen::VectorXd::Zero(b.size()),
                              relative_tolerance * b.norm()));
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
