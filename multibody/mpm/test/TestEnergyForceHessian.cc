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




}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
