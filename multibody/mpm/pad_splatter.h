#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/multibody/mpm/pad.h"
#include "drake/multibody/mpm/internal/b_spline.h"

namespace drake {
namespace multibody {
namespace mpm {


template <typename T>
class PadSplatter {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PadSplatter);



  void SplatToPad(cont T& mass, const Vector3<T>& v, Pad<T>* local_pad) {
    // for i = 0 : 27, 
    // local_pad[i] += weights[i] * mass
    // ...
  }



 private:
  std::array<size_t, 27> pad_nodes_global_indices_{};
  std::array<T, 27> weights_{};
  std::array<Vector3<T>, 27> weight_gradients_{};

};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
