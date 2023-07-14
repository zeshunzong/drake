#pragma once

#include <memory>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
// #include "drake/multibody/fem/fem_indexes.h"
// #include "drake/multibody/mpm/mpm_state_system.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/mpm/Particles.h"

#include <iostream>

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmState {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmState);

  // /** Creates an "owned" version of %FemState that allocates and accesses states
  //  and cached data using the provided `system`. The %FemState created with this
  //  constructor owns the states and data.
  //  @pre system != nullptr. */
  // explicit MpmState(const internal::MpmStateSystem<T>* system);

  // /** Creates a "shared" version of %FemState that accesses states and cached
  //  data in the given `context`. The %FemState created with this constructor
  //  doesn't own the states and data.
  //  @pre system != nullptr.
  //  @pre context != nullptr.
  //  @pre system and context are compatible. */
  // MpmState(const internal::MpmStateSystem<T>* system,
  //          const systems::Context<T>* context);

  MpmState(){

  }

  MpmState(const Particles& particles){
    particles_ = particles;
  }

  const Particles& GetParticles() const {
    return particles_;
  }

  void SetParticles(const Particles& particles) {
    particles_ = particles;
  }

  void SetState(const MpmState& mpmstate){
    particles_ = mpmstate.GetParticles();
  }


  int num_particles() const {
    return particles_.get_num_particles();
  }

  void print_info() const {
    particles_.print_info();
  }

  

  /** Returns an identical copy of `this` FemState. */
  std::unique_ptr<MpmState<T>> Clone() const;

 private:

  Particles particles_;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
