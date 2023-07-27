#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/identifier.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/mpm/mpm_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/physical_model.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/multibody/mpm/poisson_disk_sampling.h"

namespace drake {
namespace multibody {

/** Uniquely identifies a deformable body. It is valid before and after
 Finalize(). */
using DeformableBodyId = Identifier<class DeformableBodyTag>;
/** Internally indexes deformable bodies, only used after Finalize(). */
using DeformableBodyIndex = TypeSafeIndex<class DeformableBodyTag>;


/** DeformableModel implements the interface in PhysicalModel and provides the
 functionalities to specify deformable bodies. Unlike rigid bodies, the shape of
 deformable bodies can change in a simulation. Each deformable body is modeled
 as a volumetric mesh with persisting topology, changing vertex positions, and
 an approximated signed distance field. A finite element model is built for each
 registered deformable body that is used to evaluate the dynamics of the body.
 @experimental
 @tparam_double_only */
template <typename T>
class DeformableModel final : public multibody::PhysicalModel<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableModel)

  /** Constructs a DeformableModel to be owned by the given MultibodyPlant.
   @pre plant != nullptr.
   @pre Finalize() has not been called on `plant`. */
  explicit DeformableModel(MultibodyPlant<T>* plant) : plant_(plant) {
    DRAKE_DEMAND(plant_ != nullptr);
    DRAKE_DEMAND(!plant_->is_finalized());
  }

  /** Returns the number of deformable bodies registered with this
   DeformableModel. */
  int num_bodies() const { return reference_positions_.size(); }

  // TODO(xuchenhan-tri): Document the minimal requirement on the geometry
  //  instance. For example, it must have a friction proximity property to be
  //  simulated with an MbP that involves contact.
  // TODO(xuchenhan-tri): Consider allowing registering deformable bodies with
  //  non-world frames.
  /** Registers a deformable body in `this` DeformableModel with the given
   GeometryInstance. The body is represented in the world frame and simulated
   with FEM with linear elements and a first order quadrature rule that
   integrates linear functions exactly. See FemModel for details. Returns a
   unique identifier for the added geometry.
   @param[in] geometry_instance  The geometry to be registered with the model.
   @param[in] config             The physical properties of deformable body.
   @param[in] resolution_hint    The parameter that guides the level of mesh
                                 refinement of the deformable geometry. It has
                                 length units (in meters) and roughly
                                 corresponds to a typical edge length in the
                                 resulting mesh for a primitive shape.
   @pre resolution_hint > 0.
   @throws std::exception if Finalize() has been called on the multibody plant
   owning this deformable model. */
  DeformableBodyId RegisterDeformableBody(
      std::unique_ptr<geometry::GeometryInstance> geometry_instance,
      const fem::DeformableBodyConfig<T>& config, double resolution_hint);

  DeformableBodyId RegisterMpmBody(
      std::unique_ptr<mpm::AnalyticLevelSet> geometry_level_set,
      std::unique_ptr<mpm::ElastoPlasticModel<double>> constitutive_model,
      multibody::SpatialVelocity<double>& geometry_initial_veolocity,
      math::RigidTransform<double>& geometry_initial_pose, double density,
      double grid_h, int min_num_particles_per_cell);

  systems::AbstractStateIndex particles_container_index() const {
    return particles_container_index_;
  }

  bool ExistsMpmModel() const {
    return (mpm_model_!= nullptr);
  }

  // TODO(xuchenhan-tri): Consider pulling PosedHalfSpace out of internal
  // namespace and use it here.
  /** Sets wall boundary conditions for the body with the given `id`. All
   vertices of the mesh of the deformable body whose reference positions are
   inside the prescribed open half space are put under zero displacement
   boundary conditions. The open half space is defined by a plane with outward
   normal n_W. A vertex V is considered to be subject to the boundary condition
   if n̂ ⋅ p_QV < 0 where Q is a point on the plane and n̂ is normalized n_W.
   @param[in] id    The body to be put under boundary condition.
   @param[in] p_WQ  The position of a point Q on the plane in the world frame.
   @param[in] n_W   Outward normal to the half space expressed in the world
                    frame.
   @pre n_W.norm() > 1e-10.
   @warning Be aware of round-off errors in floating computations when placing a
   vertex very close to the plane defining the half space.
   @throws std::exception if Finalize() has been called on the multibody plant
   owning this deformable model or if no deformable body with the given `id` has
   been registered in this model. */
  void SetWallBoundaryCondition(DeformableBodyId id, const Vector3<T>& p_WQ,
                                const Vector3<T>& n_W);

  /** Returns the discrete state index of the deformable body identified by the
   given `id`.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
   or if no deformable body with the given `id` has been registered in this
   model. */
  systems::DiscreteStateIndex GetDiscreteStateIndex(DeformableBodyId id) const;


  /** Returns the abstract state index of the only mpm body
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
   or if mpm_model has not been registered
   model. */
  systems::AbstractStateIndex GetParticlesAbstractIndex() const{
      this->ThrowIfSystemResourcesNotDeclared(__func__);
      if (!ExistsMpmModel()) {
        throw std::logic_error("No MPM Model registered");
      }
      return mpm_model_->particles_container_index_;
  }

  /** Returns the FemModel for the body with `id`.
   @throws exception if no deformable body with `id` is registered with `this`
   %DeformableModel. */
  const fem::FemModel<T>& GetFemModel(DeformableBodyId id) const;

  /** Returns the very MPM 
   @throws exception if no mpm mpdel  */
  const mpm::MpmModel<T>& GetMpmModel() const;

  // TODO(xuchenhan-tri): The use of T over double is not well-reasoned.
  //  Consider whether T is really necessary when we support autodiff in
  //  deformable simulations.
  /** Returns the reference positions of the vertices of the deformable body
   identified by the given `id`.
   The reference positions are represented as a VectorX with 3N values where N
   is the number of vertices. The x-, y-, and z-positions (measured and
   expressed in the world frame) of the j-th vertex are 3j, 3j + 1, and 3j + 2
   in the VectorX.
   @throws std::exception if no deformable body with the given `id` has been
   registered in this model. */
  const VectorX<T>& GetReferencePositions(DeformableBodyId id) const;

  /** Returns the DeformableBodyId of the body with the given body index.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet
   or if index is larger than or equal to the total number of registered
   deformable bodies. */
  DeformableBodyId GetBodyId(DeformableBodyIndex index) const;

  /** (Internal) Returns the DeformableBodyIndex of the body with the given id.
   This function is for internal bookkeeping use only. Most users should use
   DeformableBodyId instead.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet
   or if no body with the given `id` has been registered. */
  DeformableBodyIndex GetBodyIndex(DeformableBodyId id) const;

  /** Returns the GeometryId of the geometry associated with the body with the
   given `id`.
   @throws std::exception if no body with the given `id` has been registered. */
  geometry::GeometryId GetGeometryId(DeformableBodyId id) const;

  /** Returns the DeformableBodyId associated with the given `geometry_id`.
   @throws std::exception if the given `geometry_id` does not correspond to a
   deformable body registered with this model. */
  DeformableBodyId GetBodyId(geometry::GeometryId geometry_id) const;

  /** Returns the output port of the vertex positions for all registered
   deformable bodies.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
  */
  const systems::OutputPort<T>& vertex_positions_port() const {
    this->ThrowIfSystemResourcesNotDeclared(__func__);
    return plant_->get_output_port(vertex_positions_port_index_);
  }


  /** Returns the output port of the mpm particle positions for the mpm body
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
  */
  const systems::OutputPort<T>& mpm_particle_positions_port() const {
    this->ThrowIfSystemResourcesNotDeclared(__func__);
    if (mpm_model_== nullptr){
      throw std::logic_error("vertex_positions_port(): No MPM Model registered");
    }
    return plant_->get_output_port(mpm_particle_positions_port_index_);
  }

  mpm::KinematicCollisionObjects<double> collision_objects_{};

 private:
  PhysicalModelPointerVariant<T> DoToPhysicalModelPointerVariant()
      const final {
    return PhysicalModelPointerVariant<T>(this);
  }

  // TODO(xuchenhan-tri): Implement CloneToDouble() and CloneToAutoDiffXd()
  // and the corresponding is_cloneable methods.

  void DoDeclareSystemResources(MultibodyPlant<T>* plant) final;

  /* Builds a FEM model for the body with `id` with linear tetrahedral elements
   and a single quadrature point. The reference positions as well as the
   connectivity of the elements are given by `mesh`, and physical properties
   such as the material model of the body are given by `config`.
   @throws exception if an FEM model corresponding to `id` already exists. */
  void BuildLinearVolumetricModel(DeformableBodyId id,
                                  const geometry::VolumeMesh<double>& mesh,
                                  const fem::DeformableBodyConfig<T>& config);

 /* Builds a MPM model for the body with `id`, basically call the MPMModel's builder
 and add the newly built mpm_model to class attribute */
  void BuildMpmModel(DeformableBodyId id,
                                  const fem::DeformableBodyConfig<T>& config,
                                  const mpm::Particles<double>& particles);

  template <template <class, int> class Model>
  void BuildLinearVolumetricModelHelper(
      DeformableBodyId id, const geometry::VolumeMesh<double>& mesh,
      const fem::DeformableBodyConfig<T>& config);

  /* Copies the vertex positions of all deformable bodies to the output port
   value which is guaranteed to be of type GeometryConfigurationVector. */
  void CopyVertexPositions(const systems::Context<T>& context,
                           AbstractValue* output) const;
  void CopyMpmPositions(const systems::Context<T>& context,
                           AbstractValue* output) const;


    // Initialize particles' positions with Poisson disk sampling. The object's
  // level set in the physical frame is the given level set in the reference
  // frame transformed by pose. We assume every particles have equal reference
  // volumes, then we can initialize particles' masses with the given constant
  // density, Finally, we initialize the velocities of particles with the
  // constant given velocity.
  void InitializeParticles(const mpm::AnalyticLevelSet& level_set,
                            const math::RigidTransform<double>& pose,
                            const typename mpm::MpmModel<T>::MaterialParameters& m_param, double grid_h,
                            mpm::Particles<double>& particles);    

  void InitializeParticles(std::string obj_filename, const math::RigidTransform<double>& pose,
                            const typename mpm::MpmModel<T>::MaterialParameters& m_param,
                            mpm::Particles<double>& particles);                     

  /* Helper to throw a useful message if a deformable body with the given `id`
   doesn't exist. */
  void ThrowUnlessRegistered(const char* source_method,
                             DeformableBodyId id) const;

  /* The MultibodyPlant that owns `this` DeformableModel. */
  MultibodyPlant<T>* plant_{nullptr};
  /* The positions of each vertex of deformable body at reference configuration.
   */
  std::unordered_map<DeformableBodyId, VectorX<T>> reference_positions_;
  /* The discrete state indexes for all deformable bodies. */
  std::unordered_map<DeformableBodyId, systems::DiscreteStateIndex>
      discrete_state_indexes_;
  std::unordered_map<DeformableBodyId, geometry::GeometryId>
      body_id_to_geometry_id_;
  std::unordered_map<geometry::GeometryId, DeformableBodyId>
      geometry_id_to_body_id_;
  std::unordered_map<DeformableBodyId, std::unique_ptr<fem::FemModel<T>>>
      fem_models_;
  std::vector<DeformableBodyId> body_ids_;
  std::unordered_map<DeformableBodyId, DeformableBodyIndex> body_id_to_index_;
  systems::OutputPortIndex vertex_positions_port_index_;
  

    // for mpm only, assume only one mpm body
  std::unique_ptr<mpm::MpmModel<T>> mpm_model_;
  systems::AbstractStateIndex particles_container_index_;
  systems::OutputPortIndex mpm_particle_positions_port_index_;

};

}  // namespace multibody
}  // namespace drake
