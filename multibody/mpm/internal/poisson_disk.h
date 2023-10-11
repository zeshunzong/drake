#pragma once

#include <array>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

std::vector<Vector3<double>> PoissonDiskSampling(
    double r, const std::array<double, 3>& x_min,
    const std::array<double, 3>& x_max);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
