#pragma once

#include <array>
#include <memory>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct Pad {
  std::array<T, 27> masses{};
  std::array<Vector3<T>, 27> momentums{};
  std::array<Vector3<T>, 27> forces{};
  Vector3<int> base_node{};
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
