#pragma once

#include <array>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/**
 * Returns a std::vector of point positions in 3D. Those randomly chosen points
 * fill in the bounding box described by [xmin[0], xmax[0]]X[xmin[1],
 * xmax[1]]X[xmin[2], xmax[2]], while maintaining a minimum distance of r
 * between each other.
 * See https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf.
 */
std::vector<Vector3<double>> PoissonDiskSampling(
    double r, const std::array<double, 3>& x_min,
    const std::array<double, 3>& x_max);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
