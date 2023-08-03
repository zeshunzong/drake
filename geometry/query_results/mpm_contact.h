#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
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

std::vector<int> particles_in_contact_:                      8             9                12
std::vector<GeometryId> participating_rigid_bodies_:        id B          id A             id B
std::vector<T> penetration_distances_:                   -dist(8, B)    -dist(9, A)      -dist(12, B)
std::vector<Vector3<T>> normal_vectors_:               unit vectors in the above directions



std::unordered_map<GeometryId, std::vector<int>> map_geometries_to_contact_particles_: map(A) = [9], map(B) = [8,12]
std::unordered_map<int, T> map_particle_index_to_distance_:  map(8) = -dist(8, B)
std::unordered_map<int, std::vector3<T>> map_particle_index_to_normal_
 */
template <typename T>
class MpmContact {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmContact)

  MpmContact() = default;


  void Reset() {
   map_geometries_to_contact_particles_.clear();
   map_particle_index_to_distance_.clear();
   map_particle_index_to_normal_.clear();
  }

  void AddContactParticle(int particle_index, GeometryId geometry_in_contact, T distance, const Vector3<T> contact_normal){

      AddContactParticleToGeometryMap(particle_index, geometry_in_contact);
      map_particle_index_to_distance_.insert({particle_index, distance});
      // map_particle_index_to_normal_.insert()
  }

   

//   const std::vector<DeformableContactSurface<T>>& contact_surfaces() const {
//     return contact_surfaces_;
//   }

//   /* Returns the contact participating information of the deformable geometry
//    with the given id.
//    @pre A geometry with `deformable_id` has been registered via
//    RegisterDeformableGeometry(). */
//   const ContactParticipation& contact_participation(
//       GeometryId deformable_id) const {
//     return contact_participations_.at(deformable_id);
//   }






  

 private:

   void AddContactParticleToGeometryMap(int particle_index, GeometryId geometry_in_contact){
      if (map_geometries_to_contact_particles_.count(geometry_in_contact) == 0) {
         std::vector<int> particles_in_contact_with_this_geometry{};
         particles_in_contact_with_this_geometry.push_back(particle_index);
         map_geometries_to_contact_particles_.insert({geometry_in_contact, particles_in_contact_with_this_geometry});
      }
      else {
         std::vector<int>& particles_in_contact_with_this_geometry = map_geometries_to_contact_particles_[geometry_in_contact];
         particles_in_contact_with_this_geometry.push_back(particle_index);
      }
   }

   std::unordered_map<GeometryId, std::vector<int>> map_geometries_to_contact_particles_{};
   std::unordered_map<int, T> map_particle_index_to_distance_{};
   std::unordered_map<int, Vector3<T>> map_particle_index_to_normal_{};

//   std::vector<int> particles_in_contact_;
//   std::vector<GeometryId> participating_rigid_bodies_;
//   std::vector<Vector3<T>> normal_vectors_;
//   std::vector<T> penetration_distances_;
  
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
