#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/plant/contact_pair_kinematics.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/discrete_update_manager.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/mpm/mpm_solver.h"

#include "drake/geometry/query_results/mpm_contact.h"


namespace drake {
namespace multibody {
namespace internal {

/* Helper class for DeformableDriver that acts both as a multiplexer and a
 demultiplexer -- it combines multiple Eigen vectors into a single stacked
 vector and it also splits an Eigen vector into multiple vectors.
 @tparam_default_scalar */
template <typename T>
class Multiplexer {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Multiplexer);

  /* Create an invalid Multiplexer. It cannot be used to (de)multiplex any
   vectors. */
  Multiplexer() = default;

  /* Constructs a Multiplexer that combines and splits vectors of the given
   sizes.
   @pre `sizes` is not empty and each entry is non-negative. */
  explicit Multiplexer(std::vector<int> sizes);

  /* The number of vectors to be multiplexed. */
  int num_vectors() const { return sizes_.size(); }

  /* Combines the given vectors into a single vector.
   @throws std::exception if the sizes of `inputs` aren't compatible with the
   sizes provided at construction. */
  VectorX<T> Multiplex(std::vector<VectorX<T>>&& inputs) const;

  /* Splits the given vector into multiple vectors and returns the one with
   the given `index`.
   @throws std::exception if the size of `input` is not the sum of sizes
   provided at construction.
   @throws std::exception if index is not in [0, num_vectors).
   @returns a vector block of the indexed vector. */
  Eigen::Ref<const VectorX<T>> Demultiplex(
      const Eigen::Ref<const VectorX<T>>& input, int index) const;

  /* Mutable version of `Demultiplex()` that takes a pointer to a stacked
   vector. */
  Eigen::Ref<VectorX<T>> Demultiplex(EigenPtr<VectorX<T>> input,
                                     int index) const;

 private:
  std::vector<int> sizes_;
  std::vector<int> offsets_;
  /* The sum over `sizes_`. */
  int num_entries_{0};
};

/* DeformableDriver is responsible for computing dynamics information about
 all deformable bodies. It works in tandem with a DeformableModel and a
 DiscreteUpdateManager that are provided at construction time. The deformable
 model informs the driver of modeling choices of the deformable bodies
 such as its Finite Element Model. The discrete update manager consumes the
 results of the computation performed by the driver and also provides
 information about the result of the world that the deformable bodies are
 interacting with. In particular, the manager provides access to MultibodyPlant.

 For any vertex in a deformable body, we say that it is "participating in
 contact and constraints" (or "participating" in short) if it is incident to a
 tetrahedron containing one or more contact points or explicitly specified as
 under constraint. We say a degree of freedom (dof) is "participating" if it
 belongs to a participating vertex. DeformableDriver reports participating
 quantities in increasing order of deformable body indexes. That is, the
 participating quantities of body 0 come first, followed by participating
 quantities of body 1 and so on. Within a single body, the participating
 vertices/dofs are ordered according to their associated vertex indexes.

 @tparam_double_only */
template <typename T>
class DeformableDriver : public ScalarConvertibleComponent<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableDriver)

  /* Constructs a deformable driver that solves for the dynamics of the given
   `deformable_model`. The newly constructed driver is used in the given
   `manager` to perform discrete updates. The given `deformable_model` and
   `manager` must outlive this driver.
   @pre deformable_model != nullptr.
   @pre manager != nullptr. */
  DeformableDriver(const DeformableModel<T>* deformable_model,
                   const DiscreteUpdateManager<T>* manager);

  ~DeformableDriver();

  // it should be that contact detection happens after free motion mpm solve
  // so in contact detection, we have already had ordered particles and grid nodes.
  // here we just do the ordering and compute weights, for testing purpose
  void MakeGridCompatibleWithParticleTestPurpose(systems::Context<T>* context) {
    
  }

  int num_deformable_bodies() const { return deformable_model_->num_bodies(); }

  // TODO(xuchenhan-tri): Implement CloneToDouble() and allow cloning to double.
  bool is_cloneable_to_double() const final { return false; }
  bool is_cloneable_to_autodiff() const final { return false; }
  bool is_cloneable_to_symbolic() const final { return false; }

  /* Declares cache entries used by this DeformableDriver through the given
   manager.
   @pre `manager` is not nullptr and points to the same DiscreteUpdateManager
   provided at construction. */
  void DeclareCacheEntries(DiscreteUpdateManager<T>* manager);

  /* Evaluates the velocities of all participating dofs. See class documentation
   for how the velocities are ordered. */
  const VectorX<T>& EvalParticipatingVelocities(
      const systems::Context<T>& context) const;

  /* Evaluates the free motion velocities of all participating dofs. See class
   documentation for how the velocities are ordered. */
  const VectorX<T>& EvalParticipatingFreeMotionVelocities(
      const systems::Context<T>& context) const;

  /* Appends the linear dynamics matrices for participating dofs of each
   deformable body registered in this model to `A` in increasing order of
   deformable body indexes. The matrix corresponding to a body without any
   participating dof is empty.
   @pre A != nullptr. */
  void AppendLinearDynamicsMatrix(const systems::Context<T>& context,
                                  std::vector<MatrixX<T>>* A) const;

  /* Given the configuration stored in `context`, appends discrete pairs in
   which one of the body in contact is deformable to the given `pairs`.
   @pre pairs != nullptr. */
  void AppendDiscreteContactPairs(
      const systems::Context<T>& context,
      std::vector<DiscreteContactPair<T>>* pairs) const;

    /* Given the configuration stored in `context`, appends the mpm part. 
    This function should be called by the above function
   @pre pairs != nullptr. */
  void AppendDiscreteContactPairsMpm(
      const systems::Context<T>& context,
      std::vector<DiscreteContactPair<T>>* pairs) const;

  void AppendDiscreteContactPairsMpm(
        const geometry::QueryObject<T>& query_object, 
        const drake::multibody::mpm::Particles<T>& particles, 
        std::vector<DiscreteContactPair<T>>* result) const;

  /* Appends the contact kinematics information for each contact pair where at
   least one of the body in contact is deformable.
   @pre result != nullptr. */
  void AppendContactKinematics(
      const systems::Context<T>& context,
      std::vector<ContactPairKinematics<T>>* result) const;

  // note: the particles that join the computation comes directly from evaluating the context
  void AppendContactKinematicsMpm(
      const systems::Context<T>& context,
      std::vector<ContactPairKinematics<T>>* result) const;

  // note: in this implementation, mpm data are directly obtained via particles
  // context is used for the rigid part
  // this allows external manipulation of particles
  void AppendContactKinematicsMpm(const systems::Context<T>& context,
    const geometry::QueryObject<T>& query_object, 
    const drake::multibody::mpm::Particles<T>& particles,
    std::vector<ContactPairKinematics<T>>* result) const;

  /* Evaluates FemState at the next time step for each deformable body and
   copies the them into the corresponding DiscreteValues.
   @pre next_states != nullptr. */
  void CalcDiscreteStates(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const;

  /* Evaluates FemState at the next time step for each deformable body and
   copies the them into the corresponding DiscreteValues.
   @pre next_states != nullptr. */
  void CalcAbstractStates(const systems::Context<T>& context,
                          systems::State<T>* update) const;

  /* Evaluates the multiplexer for participating velocities for all bodies.
   @pre result != nullptr. */
  const Multiplexer<T>& EvalParticipatingVelocityMultiplexer(
      const systems::Context<T>& context) const;

  void DummyCheckContext(const systems::Context<T>& context) const;

 private:
  friend class DeformableDriverTest;
  friend class DeformableDriverContactTest;
  friend class DeformableDriverContactKinematicsTest;
  friend class DeformableIntegrationTest;

  /* Struct used to conglomerate the indexes of cache entries declared by
   the manager. */
  struct CacheIndexes {
    /* Per body cache entries indexed by DeformableBodyIndex. */
    std::vector<systems::CacheIndex> fem_states;
    std::vector<systems::CacheIndex> free_motion_fem_states;
    std::vector<systems::CacheIndex> next_fem_states;

    // we assume there is only one MPM model, so no need for a vector
    systems::CacheIndex mpm_state;
    systems::CacheIndex free_motion_mpm_state;
    systems::CacheIndex next_mpm_state;
    systems::CacheIndex mpm_solver_scratch;


    std::vector<systems::CacheIndex> fem_solver_scratches;
    systems::CacheIndex deformable_contact;
    std::vector<systems::CacheIndex> dof_permutations;
    std::unordered_map<geometry::GeometryId, systems::CacheIndex>
        vertex_permutations;
    systems::CacheIndex participating_velocity_mux;
    systems::CacheIndex participating_velocities;
    systems::CacheIndex participating_free_motion_velocities;
    std::vector<systems::CacheIndex> free_motion_tangent_matrices;
    std::vector<systems::CacheIndex>
        free_motion_tangent_matrix_schur_complements;
  };
  /* Copies the state of the deformable body with `id` in the given `context`
   to the `fem_state`.
   @pre fem_state != nullptr and has size compatible with the state of the
        deformable body with the given `index`.
   @pre `index` is valid and less than the number of deformable bodies. */
  void CalcFemState(const systems::Context<T>& context,
                    DeformableBodyIndex index,
                    fem::FemState<T>* fem_state) const;

  /* Eval version of CalcFemState(). */
  const fem::FemState<T>& EvalFemState(const systems::Context<T>& context,
                                       DeformableBodyIndex index) const;

  const mpm::MpmState<T>& EvalMpmState(const systems::Context<T>& context) const;

  void CalcMpmState(const systems::Context<T>& context, mpm::MpmState<T>* mpm_state) const;
                                     

  /* Given the state of the deformable body with `index` in the given `context`,
   computes its "free motion" state (the state the body would have at the next
   time step in the absence of contact or constraints).
   @pre fem_state_star != nullptr and is compatible with the state of the
   deformable body with the given `index`. */
  void CalcFreeMotionFemState(const systems::Context<T>& context,
                              DeformableBodyIndex index,
                              fem::FemState<T>* fem_state_star) const;

  /* Eval version of CalcFreeMotionFemState(). */
  const fem::FemState<T>& EvalFreeMotionFemState(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `index` in the given `context`,
   computes its "free motion" state (the state the body would have at the next
   time step in the absence of contact or constraints).
   @pre fem_state_star != nullptr and is compatible with the state of the
   deformable body with the given `index`. */
  void CalcFreeMotionMpmState(const systems::Context<T>& context,
                              mpm::MpmState<T>* mpm_state_star) const;

  /* Eval version of CalcFreeMotionFemState(). */
  const mpm::MpmState<T>& EvalFreeMotionMpmState(
      const systems::Context<T>& context) const;

  /* Given the state of the deformable body with `index` in the given `context`,
   computes the state of the deformable body at the next time step.
   @note The state of the deformable body will the same as the "free motion"
         state in the absence of contact or constraints. Otherwise, the discrete
         solver results for participating dofs are evaluated, and the Schur
         complement of the tangent matrix is used to update the
         non-participating dofs.
   @pre next_fem_state != nullptr and is compatible with the state of
        the deformable body with the given `index`. */
  void CalcNextFemState(const systems::Context<T>& context,
                        DeformableBodyIndex index,
                        fem::FemState<T>* next_fem_state) const;

  /* Eval version of CalcNextFemState(). */
  const fem::FemState<T>& EvalNextFemState(const systems::Context<T>& context,
                                           DeformableBodyIndex index) const;
  /* Eval version of CalcNextMpmState(). */
  const mpm::MpmState<T>& EvalNextMpmState(const systems::Context<T>& context) const;
    void CalcNextMpmState(const systems::Context<T>& context,
                            mpm::MpmState<T>* next_mpm_state) const;
  /* Computes the contact information for all registered deformable bodies
   @pre The geometry query input port of the MultibodyPlant that owns the
        manager associated with this DeformableDriver is connected.
   @pre result != nullptr. */
  void CalcDeformableContact(
      const systems::Context<T>& context,
      geometry::internal::DeformableContact<T>* result) const;

  void CalcMpmContact(
    const geometry::QueryObject<T>& query_object, 
    const drake::multibody::mpm::Particles<T>& current_particles, 
    geometry::internal::MpmContact<T>* result) const;

  /* Eval version of CalcDeformableContact(). */
  const geometry::internal::DeformableContact<T>& EvalDeformableContact(
      const systems::Context<T>& context) const;

  /* Computes the partial permutation that maps degrees of freedom of the
   deformable body with the given `index` to degrees of freedom that belong to
   vertices of the body that participate in contact.
   @pre result != nullptr. */
  void CalcDofPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index,
      contact_solvers::internal::PartialPermutation* result) const;

  /* Eval version of CalcDofPermutation(). */
  const contact_solvers::internal::PartialPermutation& EvalDofPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Computes the partial permutation that maps vertices of the
   deformable geometry with the given `id` to vertices that belong to
   vertices of the geometry that participate in contact.
   @pre result != nullptr. */
  void CalcVertexPermutation(
      const systems::Context<T>& context, geometry::GeometryId id,
      contact_solvers::internal::PartialPermutation* result) const;

  /* Eval version of CalcVertexPermutation(). */
  const contact_solvers::internal::PartialPermutation& EvalVertexPermutation(
      const systems::Context<T>& context, geometry::GeometryId id) const;

  /* Calc version of EvalParticipatingVelocityMultiplexer(). */
  void CalcParticipatingVelocityMultiplexer(const systems::Context<T>& context,
                                            Multiplexer<T>* result) const;

  /* Calc version of EvalParticipatingVelocities().
   @pre result != nullptr. */
  void CalcParticipatingVelocities(const systems::Context<T>& context,
                                   VectorX<T>* result) const;

  /* Calc version of EvalParticipatingFreeMotionVelocities().
   @pre result != nullptr. */
  void CalcParticipatingFreeMotionVelocities(const systems::Context<T>& context,
                                             VectorX<T>* result) const;

  /* Computes the tangent matrix of the momentum balance equation for the
   deformable body with the given `index` at the free motion state.
   @pre tangent_matrix != nullptr. */
  void CalcFreeMotionTangentMatrix(
      const systems::Context<T>& context, DeformableBodyIndex index,
      fem::internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const;

  /* Eval version of CalcFreeMotionTangentMatrix(). */
  const fem::internal::PetscSymmetricBlockSparseMatrix&
  EvalFreeMotionTangentMatrix(const systems::Context<T>& context,
                              DeformableBodyIndex index) const;

  /* Computes the Schur complement of the tangent matrix of the deformable body
   with the given `index` at the free motion state (see
   EvalFreeMotionTangentMatrix()) based on contact participation. The dofs not
   participating in contact are eliminated in favor of those that do
   participate in contact. If no dof is participating, `result` is set to be
   empty and invalid for efficiency.
   @pre result != nullptr. */
  void CalcFreeMotionTangentMatrixSchurComplement(
      const systems::Context<T>& context, DeformableBodyIndex index,
      fem::internal::SchurComplement<T>* result) const;

  /* Eval version of CalcFreeMotionTangentMatrixSchurComplement(). */
  const fem::internal::SchurComplement<T>&
  EvalFreeMotionTangentMatrixSchurComplement(const systems::Context<T>& context,
                                             DeformableBodyIndex index) const;

  CacheIndexes cache_indexes_;
  /* Modeling information about all deformable bodies. */
  const DeformableModel<T>* const deformable_model_;
  const DiscreteUpdateManager<T>* const manager_;
  /* The integrator used to advance deformable body free motion states in
   time. */
  std::unique_ptr<fem::internal::DiscreteTimeIntegrator<T>> integrator_;

  /* The integrator used to advance deformable body free motion states in
   time. */
  std::unique_ptr<mpm::internal::MpmSolver<T>> mpm_solver_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
