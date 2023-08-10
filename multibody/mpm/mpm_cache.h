#pragma once

#include <memory>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/mpm/Particles.h"
#include "drake/multibody/mpm/SparseGrid.h"

#include <iostream>

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmCache {
 public:
  // DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmState);


  MpmCache(){
  }

  MpmCache(const Particles<T>& particles_in, const SparseGrid<T>& grid_in){
    particles_sorted_ = particles_in;
    compatible_grid_ = grid_in;
  }

  Particles<T> particles_sorted_{};
  SparseGrid<T> compatible_grid_{};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
