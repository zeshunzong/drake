#pragma once

#include <limits>
#include <utility>
#include <vector>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/Grid.h"

namespace drake {
namespace multibody {
namespace mpm {


void WriteGrid2obj(const std::string& filename, Grid& grid);

void WriteGridVelocity2obj(const std::string& filename, Grid& grid);

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

