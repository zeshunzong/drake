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




std::unordered_map<GeometryId, std::vector<int>> map_geometries_to_contact_particles_: map(A) = [9], map(B) = [8,12]
std::unordered_map<int, T> map_particle_index_to_distance_:  map(8) = -dist(8, B)
std::unordered_map<int, std::vector3<T>> map_particle_index_to_normal_
 */
template <typename T>
class MpmContact {
 public:


  struct MpmParticleContactPair{

   int particle_in_contact_index_{};
   GeometryId non_mpm_id_{};
   T penetration_distance_{};
   Vector3<T> normal_{};
   Vector3<T> particle_in_contact_position_{};

   MpmParticleContactPair(const int particle_in_contact_index, const GeometryId& non_mpm_id, const T penetration_distance, const Vector3<T>& normal, const Vector3<T>& position){
      particle_in_contact_index_ = particle_in_contact_index;
      non_mpm_id_ = non_mpm_id;
      penetration_distance_ = penetration_distance;
      normal_ = normal;
      particle_in_contact_position_ = position;
   }


  };
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmContact)

  MpmContact() = default;


  void Reset() {
   mpm_contact_pairs_.clear();
   particles_in_contact_.clear();
  }

  void AddMpmContactPair(const int particle_index, const GeometryId& nonmpm_geometry, const T distance, const Vector3<T>& contact_normal, const Vector3<T>& position) {
   MpmParticleContactPair new_pair(particle_index, nonmpm_geometry, distance, contact_normal, position);
   mpm_contact_pairs_.push_back(new_pair);
   
   particles_in_contact_.insert(particle_index); // mark this particle as a contact particle
  }

  size_t GetNumContactPairs() const {
   return mpm_contact_pairs_.size();
  }

  int GetParticleIndexAt(size_t contact_pair_index) {
   return mpm_contact_pairs_[contact_pair_index].particle_in_contact_index_;
  }

  GeometryId& GetNonMpmIdAt(size_t contact_pair_index) {
   return mpm_contact_pairs_[contact_pair_index].non_mpm_id_;
  }

  T GetPenetrationDistanceAt(size_t contact_pair_index) {
   return mpm_contact_pairs_[contact_pair_index].penetration_distance_;
  }

  Vector3<T>& GetNormalAt(size_t contact_pair_index) {
   return mpm_contact_pairs_[contact_pair_index].normal_;
  }

  Vector3<T>& GetContactPositionAt(size_t contact_pair_index) {
   return mpm_contact_pairs_[contact_pair_index].particle_in_contact_position_;
  }

  bool ParticleIsInContact(int particle_index) const {
   auto it = particles_in_contact_.find(particle_index);
   if (it != particles_in_contact_.end()){
      return true;
   }
   return false;
  }


  

 private:

   // void AddContactParticleToGeometryMap(int particle_index, GeometryId geometry_in_contact){
   //    if (map_geometries_to_contact_particles_.count(geometry_in_contact) == 0) {
   //       std::vector<int> particles_in_contact_with_this_geometry{};
   //       particles_in_contact_with_this_geometry.push_back(particle_index);
   //       map_geometries_to_contact_particles_.insert({geometry_in_contact, particles_in_contact_with_this_geometry});
   //    }
   //    else {
   //       std::vector<int>& particles_in_contact_with_this_geometry = map_geometries_to_contact_particles_[geometry_in_contact];
   //       particles_in_contact_with_this_geometry.push_back(particle_index);
   //    }
   // }

   // std::unordered_map<GeometryId, std::vector<int>> map_geometries_to_contact_particles_{};
   // std::unordered_map<int, T> map_particle_index_to_distance_{};
   // std::unordered_map<int, Vector3<T>> map_particle_index_to_normal_{};


   std::vector<MpmParticleContactPair> mpm_contact_pairs_{};

   std::unordered_set<int> particles_in_contact_{};

  
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
