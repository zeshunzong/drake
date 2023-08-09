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
  }

  void AddMpmContactPair(const int particle_index, const GeometryId& nonmpm_geometry, const T distance, const Vector3<T>& contact_normal, const Vector3<T>& position) {
   MpmParticleContactPair new_pair(particle_index, nonmpm_geometry, distance, contact_normal, position);
   mpm_contact_pairs_.push_back(new_pair);
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

 private:
   std::vector<MpmParticleContactPair> mpm_contact_pairs_{};

};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
