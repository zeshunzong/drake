#pragma once

#include <vector>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace geometry {
namespace internal {

/* Stores all info about mpm particles that are in contact with rigid bodies (defined to be 
particles that fall within rigid bodies).

                  Mpm Particles (endowed with an ordering)

            
            `1    `2    `3    `4
            
                             ---------
            `5    `6    `7   |*8
                             |      Rigid body with id B
      ----------             |
            *9 |  `10   `11  |*12
               |             ---------
               |
Rigid body with id A

*: particles in contact
`: particles not in contact


 */

template <typename T>
struct MpmParticleContactPair{

   size_t particle_in_contact_index{};
   GeometryId non_mpm_id{};
   T penetration_distance{};
   Vector3<T> normal{};
   Vector3<T> particle_in_contact_position{};

};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
