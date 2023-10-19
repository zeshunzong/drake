#pragma once

#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/internal/hashing_utils.h"
#include "drake/multibody/mpm/internal/math_utils.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct GridData {


  std::vector<T> masses_;
  std::vector<Vector3<T>> velocities_;
  std::vector<Vector3<T>> forces_;

  void Reserve(size_t capacity) {
    masses_.reserve(capacity);
    velocities_.reserve(capacity);
    forces_.reserve(capacity);
  }


  void ConvertMomentumToVelocity(){
    for (int i = 0; i < velocities_.size(); ++i) {
      velocities_[i] = velocities_[i] / masses_[i];
    }
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
