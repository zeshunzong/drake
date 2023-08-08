#include "drake/multibody/plant/deformable_driver.h"

#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/contact_solvers/contact_configuration.h"
#include "drake/multibody/fem/dirichlet_boundary_condition.h"
#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/fem/fem_solver.h"

#include "drake/multibody/mpm/mpm_model.h"

#include "drake/multibody/mpm/Particles.h"
#include "drake/multibody/fem/velocity_newmark_scheme.h"
#include "drake/multibody/plant/contact_properties.h"
#include "drake/systems/framework/context.h"
#include "drake/geometry/query_results/signed_distance_pair.h"
#include "drake/geometry/query_results/signed_distance_to_point.h"



using drake::geometry::GeometryId;
using drake::geometry::internal::ContactParticipation;
using drake::geometry::internal::DeformableContact;
using drake::geometry::internal::DeformableContactSurface;
using drake::geometry::internal::MpmContact;
using drake::multibody::contact_solvers::internal::Block3x3SparseMatrix;
using drake::multibody::contact_solvers::internal::ContactConfiguration;
using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::contact_solvers::internal::MatrixBlock;
using drake::multibody::contact_solvers::internal::PartialPermutation;
using drake::multibody::fem::FemModel;
using drake::multibody::fem::FemState;
using drake::multibody::mpm::MpmState;
using drake::multibody::mpm::MpmModel;
using drake::multibody::mpm::Particles;
using drake::multibody::mpm::internal::MpmSolver;
using drake::multibody::mpm::internal::MpmSolverScratchData;
using drake::multibody::fem::internal::DirichletBoundaryCondition;
using drake::multibody::fem::internal::FemSolver;
using drake::multibody::fem::internal::FemSolverScratchData;

using drake::multibody::fem::internal::PetscSymmetricBlockSparseMatrix;
using drake::multibody::fem::internal::SchurComplement;
using drake::systems::Context;


namespace drake {
namespace multibody {
namespace internal {

template <typename T>
void DeformableDriver<T>::DummyCheckContext(const systems::Context<T>& context) const {
    const MpmState<T>& current_mpm_state = EvalMpmState(context);
    const Particles<T>& current_particles = current_mpm_state.GetParticles(); 
    std::cout << "num particles in context " << current_particles.get_num_particles() << std::endl;
}

template <typename T>
DeformableDriver<T>::DeformableDriver(
    const DeformableModel<T>* deformable_model,
    const DiscreteUpdateManager<T>* manager)
    : deformable_model_(deformable_model), manager_(manager) {
  DRAKE_DEMAND(deformable_model != nullptr);
  DRAKE_DEMAND(manager != nullptr);
  // TODO(xuchenhan-tri): Expose the integrator as a config.
  /* Set the time integrator for advancing deformable states in time to be the
   midpoint rule, i.e., q = q₀ + δt/2 *(v₀ + v). */
  integrator_ = std::make_unique<fem::internal::VelocityNewmarkScheme<T>>(
      manager_->plant().time_step(), 1.0, 0.5);

  const MpmModel<T>& mpm_model = deformable_model_->GetMpmModel();
  mpm_solver_ = std::make_unique<mpm::internal::MpmSolver<T>>(&mpm_model,
      manager_->plant().time_step());
 
}

template <typename T>
DeformableDriver<T>::~DeformableDriver() = default;

template <typename T>
void DeformableDriver<T>::DeclareCacheEntries(
    DiscreteUpdateManager<T>* manager) {
  DRAKE_DEMAND(manager_ == manager);
  if (deformable_model_->num_bodies() > 0) {
    const auto& deformable_contact_cache_entry = manager->DeclareCacheEntry(
        "deformable contact data",
        systems::ValueProducer(this, &DeformableDriver<T>::CalcDeformableContact),
        {systems::System<T>::configuration_ticket()});
    cache_indexes_.deformable_contact =
        deformable_contact_cache_entry.cache_index();

    const auto& participating_velocity_mux_cache_entry =
        manager->DeclareCacheEntry(
            "multiplexer for participating velocities",
            systems::ValueProducer(
                this, &DeformableDriver<T>::CalcParticipatingVelocityMultiplexer),
            {deformable_contact_cache_entry.ticket()});
    cache_indexes_.participating_velocity_mux =
        participating_velocity_mux_cache_entry.cache_index();

    const auto& participating_velocities_cache_entry = manager->DeclareCacheEntry(
        "participating velocities for all bodies",
        systems::ValueProducer(this,
                                &DeformableDriver<T>::CalcParticipatingVelocities),
        {deformable_contact_cache_entry.ticket(),
        systems::System<T>::xd_ticket()});
    cache_indexes_.participating_velocities =
        participating_velocities_cache_entry.cache_index();

    const auto& participating_free_motion_velocities_cache_entry =
        manager->DeclareCacheEntry(
            fmt::format("participating free motion velocities for all bodies"),
            systems::ValueProducer(
                this,
                &DeformableDriver<T>::CalcParticipatingFreeMotionVelocities),
            {deformable_contact_cache_entry.ticket(),
            systems::System<T>::xd_ticket()});
    cache_indexes_.participating_free_motion_velocities =
        participating_free_motion_velocities_cache_entry.cache_index();
  
  // loop over all fem bodies
  for (DeformableBodyIndex i(0); i < deformable_model_->num_bodies(); ++i) {
    const DeformableBodyId id = deformable_model_->GetBodyId(i);
    const fem::FemModel<T>& fem_model = deformable_model_->GetFemModel(id);
    std::unique_ptr<fem::FemState<T>> model_state = fem_model.MakeFemState();
    /* Cache entry for current FEM state. */
    const auto& fem_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("FEM state for body with index {}", i),
        systems::ValueProducer(
            *model_state,
            std::function<void(const Context<T>&, fem::FemState<T>*)>{
                [this, i](const Context<T>& context, fem::FemState<T>* state) {
                  this->CalcFemState(context, i, state);
                }}),
        {systems::System<T>::xd_ticket()});
    cache_indexes_.fem_states.emplace_back(fem_state_cache_entry.cache_index());

    /* Cache entry for free motion FEM state. */
    const auto& free_motion_fem_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("free motion FEM state for body with index {}", i),
        systems::ValueProducer(
            *model_state,
            std::function<void(const systems::Context<T>&, fem::FemState<T>*)>{
                [this, i](const systems::Context<T>& context,
                          fem::FemState<T>* free_motion_state) {
                  this->CalcFreeMotionFemState(context, i, free_motion_state);
                }}),
        {fem_state_cache_entry.ticket()});
    cache_indexes_.free_motion_fem_states.emplace_back(
        free_motion_fem_state_cache_entry.cache_index());

    /* Cache entry for FEM state at next time step. */
    const auto& next_fem_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("FEM state for body with index {} at next time step", i),
        systems::ValueProducer(
            *model_state,
            std::function<void(const systems::Context<T>&, fem::FemState<T>*)>{
                [this, i](const systems::Context<T>& context,fem::FemState<T>* next_fem_state) 
                          {
                  this->CalcNextFemState(context, i, next_fem_state);
                }
                }
                ),
        {systems::SystemBase::all_sources_ticket()});
    cache_indexes_.next_fem_states.emplace_back(
        next_fem_state_cache_entry.cache_index());

    /* Scratch data for FEM solver. */
    FemSolverScratchData scratch(fem_model);
    const auto& scratch_entry = manager->DeclareCacheEntry(
        fmt::format("FEM solver scratch workspace for body with index {}", i),
        systems::ValueProducer(scratch, &systems::ValueProducer::NoopCalc),
        {systems::SystemBase::nothing_ticket()});
    cache_indexes_.fem_solver_scratches.emplace_back(
        scratch_entry.cache_index());

    /* Cache entry for the tangent matrix at free motion state. */
    std::unique_ptr<PetscSymmetricBlockSparseMatrix> model_tangent_matrix =
        fem_model.MakePetscSymmetricBlockSparseTangentMatrix();
    model_tangent_matrix->AssembleIfNecessary();
    const auto& free_motion_tangent_matrix_cache_entry =
        manager->DeclareCacheEntry(
            fmt::format("free motion tangent matrix for body with index {}", i),
            systems::ValueProducer(
                *model_tangent_matrix,
                std::function<void(const systems::Context<T>&,
                                   PetscSymmetricBlockSparseMatrix*)>{
                    [this, i](const systems::Context<T>& context,
                              PetscSymmetricBlockSparseMatrix* tangent_matrix) {
                      this->CalcFreeMotionTangentMatrix(context, i,
                                                        tangent_matrix);
                    }}),
            {free_motion_fem_state_cache_entry.ticket()});
    cache_indexes_.free_motion_tangent_matrices.emplace_back(
        free_motion_tangent_matrix_cache_entry.cache_index());

    const auto& schur_complement_cache_entry = manager->DeclareCacheEntry(
        fmt::format("free motion tangent matrix Schur complement for body "
                    "with index {}",
                    i),
        systems::ValueProducer(std::function<void(const systems::Context<T>&,
                                                  SchurComplement<T>*)>{
            [this, i](const systems::Context<T>& context,
                      SchurComplement<T>* schur_complement) {
              this->CalcFreeMotionTangentMatrixSchurComplement(
                  context, i, schur_complement);
            }}),
        {free_motion_tangent_matrix_cache_entry.ticket(),
         deformable_contact_cache_entry.ticket()});
    cache_indexes_.free_motion_tangent_matrix_schur_complements.emplace_back(
        schur_complement_cache_entry.cache_index());

    /* Permutation for participating dofs for each body. */
    const auto& dof_permutation_cache_entry = manager->DeclareCacheEntry(
        fmt::format("partial permutation for dofs of body {} based on "
                    "participation in contact",
                    i),
        systems::ValueProducer(
            std::function<void(const Context<T>&, PartialPermutation*)>{
                [this, i](const Context<T>& context,
                          PartialPermutation* result) {
                  this->CalcDofPermutation(context, i, result);
                }}),
        {deformable_contact_cache_entry.ticket()});
    cache_indexes_.dof_permutations.emplace_back(
        dof_permutation_cache_entry.cache_index());

    /* Permutation for participating vertices for each body. */
    const GeometryId g_id =
        deformable_model_->GetGeometryId(deformable_model_->GetBodyId(i));
    const auto& vertex_permutation_cache_entry = manager->DeclareCacheEntry(
        fmt::format("partial permutation for vertices of body {} based on "
                    "participation in contact",
                    i),
        systems::ValueProducer(
            std::function<void(const Context<T>&, PartialPermutation*)>{
                [this, g_id](const Context<T>& context,
                             PartialPermutation* result) {
                  this->CalcVertexPermutation(context, g_id, result);
                }}),
        {deformable_contact_cache_entry.ticket()});
    cache_indexes_.vertex_permutations.emplace(
        g_id, vertex_permutation_cache_entry.cache_index());
  } // finish all fem bodies


  }

  if (mpm_solver_ != nullptr) {
    /* Scratch data for MpmSolver. */

    if (deformable_model_->ExistsMpmModel()){ 

        MpmSolverScratchData<T> scratch(deformable_model_->GetMpmModel().grid_h_);
        const auto& scratch_entry = manager->DeclareCacheEntry(
            fmt::format("MpmSolver scratch"),
            systems::ValueProducer(scratch, &systems::ValueProducer::NoopCalc),
            {systems::SystemBase::nothing_ticket()});
        cache_indexes_.mpm_solver_scratch = scratch_entry.cache_index();

    } else { throw;}
  }
  // cache mpm, if there exists one
  if (deformable_model_->ExistsMpmModel()){
    const mpm::MpmModel<T>& mpm_model = deformable_model_->GetMpmModel();
    std::unique_ptr<mpm::MpmState<T>> model_state = mpm_model.MakeMpmState();
    //---------------------------- cache current state--------------------
    const auto& mpm_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("MPM state"),
        systems::ValueProducer(
            *model_state,
            std::function<void(const Context<T>&, mpm::MpmState<T>*)>{
                [this](const Context<T>& context, mpm::MpmState<T>* state) {
                  this->CalcMpmState(context, state);
                }}),
        {systems::SystemBase::all_sources_ticket()}); // TODO: add prereq dependency
    cache_indexes_.mpm_state = mpm_state_cache_entry.cache_index();

    /* Cache entry for free motion MPM state. */
    const auto& free_motion_mpm_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("free motion MPM state"),
        systems::ValueProducer(
            *model_state,
            std::function<void(const systems::Context<T>&, mpm::MpmState<T>*)>{
                [this](const systems::Context<T>& context,
                          mpm::MpmState<T>* free_motion_state) {
                  this->CalcFreeMotionMpmState(context, free_motion_state);
                }}),
        {systems::SystemBase::all_sources_ticket()}); // TODO: add prereq dependency
    cache_indexes_.free_motion_mpm_state = free_motion_mpm_state_cache_entry.cache_index();


    /* Cache entry for MPM state at next time step. */
    const auto& next_mpm_state_cache_entry = manager->DeclareCacheEntry(
        fmt::format("MPM state at next time step"),
        systems::ValueProducer(
            *model_state,
            std::function<void(const systems::Context<T>&, mpm::MpmState<T>*)>{
                [this](const systems::Context<T>& context,
                          mpm::MpmState<T>* next_mpm_state) {
                  this->CalcNextMpmState(context, next_mpm_state);
                }}),
        {systems::SystemBase::all_sources_ticket()});
    cache_indexes_.next_mpm_state = next_mpm_state_cache_entry.cache_index();

  }


}

template <typename T>
void DeformableDriver<T>::AppendLinearDynamicsMatrix(
    const systems::Context<T>& context, std::vector<MatrixX<T>>* A) const {
  DRAKE_DEMAND(A != nullptr);
  const int num_bodies = deformable_model_->num_bodies();
  for (DeformableBodyIndex index(0); index < num_bodies; ++index) {
    const SchurComplement<T>& schur_complement =
        EvalFreeMotionTangentMatrixSchurComplement(context, index);
    A->emplace_back(schur_complement.get_D_complement());
  }
}

template <typename T>
void DeformableDriver<T>::AppendDiscreteContactPairs(
    const systems::Context<T>& context,
    std::vector<DiscreteContactPair<T>>* result) const {
  DRAKE_DEMAND(result != nullptr);
  std::vector<DiscreteContactPair<T>>& contact_pairs = *result;

  const geometry::QueryObject<T>& query_object =
      manager_->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();
  const DeformableContact<T>& deformable_contact =
      EvalDeformableContact(context);

  for (const DeformableContactSurface<double>& surface :
       deformable_contact.contact_surfaces()) {
    /* While our discrete solvers might model constraints as compliant, an
    infinite stiffness indicates to use the stiffest approximation possible
    without sacrifycing numerical conditioning. SAP will use the "near rigid"
    regime approximation in this case. */
    const T k = std::numeric_limits<double>::infinity();

    // TODO(xuchenhan-tri): Currently, body_B is guaranteed to be
    // non-deformable. When we support deformable vs. deformable contact, we
    // need to update this logic for retrieving body names.
    DRAKE_DEMAND(manager_->geometry_id_to_body_index().count(surface.id_B()) >
                 0);
    // TODO(xuchenhan-tri): Currently deformable bodies don't have names. When
    // they do get names upon registration (in DeformableModel), update its body
    // name here.
    const std::string body_A_name(
        fmt::format("deformable body with geometry id {}", surface.id_A()));
    const BodyIndex body_B_index =
        manager_->geometry_id_to_body_index().at(surface.id_B());
    const Body<T>& body_B = manager_->plant().get_body(body_B_index);
    /* We use dt as the default dissipation constant so that the contact is in
     near-rigid regime and the compliance is only used as stabilization. */
    const T tau = GetCombinedDissipationTimeConstant(
        surface.id_A(), surface.id_B(), manager_->plant().time_step(),
        body_A_name, body_B.name(), inspector);
    const double mu = GetCombinedDynamicCoulombFriction(
        surface.id_A(), surface.id_B(), inspector);

    for (int i = 0; i < surface.num_contact_points(); ++i) {
      const Vector3<T>& p_WC = surface.contact_points_W()[i];
      const Vector3<T>& nhat_BA_W = surface.nhats_W()[i];
      const T& phi0 = surface.signed_distances()[i];
      const T fn0 = NAN;  // not used.
      const T d = NAN;    // not used.
      contact_pairs.push_back({-1, surface.id_A(), surface.id_B(), p_WC, nhat_BA_W,
                               phi0, fn0, k, d, tau, mu});
    }
  }
}

template <typename T>
void DeformableDriver<T>::AppendDiscreteContactPairsMpm(const systems::Context<T>& context,std::vector<DiscreteContactPair<T>>* result) const {

    MpmContact<T> mpm_contact;
    CalcMpmContact(context, &mpm_contact); // compute all mpm contact pairs
    GeometryId mpm_id = GeometryId::get_new_id(); // don't know how to use
    DRAKE_DEMAND(result != nullptr);
    std::vector<DiscreteContactPair<T>>& contact_pairs = *result;
    for (size_t i = 0; i < mpm_contact.GetNumContactPairs(); i++) {

        const T fn0 = NAN; const T d = NAN;  // not used.
        const T k = std::numeric_limits<double>::infinity();
        const T tau = 0.5; const T mu = 0.5;
        mpm_contact.GetParticleIndexAt(i);
        mpm_contact.GetNonMpmIdAt(i);
        mpm_contact.GetPenetrationDistanceAt(i);
        mpm_contact.GetNormalAt(i);

        contact_pairs.push_back({mpm_contact.GetParticleIndexAt(i),
                                 mpm_id, 
                                 mpm_contact.GetNonMpmIdAt(i), 
                                 mpm_contact.GetContactPositionAt(i),
                                 mpm_contact.GetNormalAt(i),
                                mpm_contact.GetPenetrationDistanceAt(i), fn0, k, d, tau, mu});
    }
}


template <typename T>
void DeformableDriver<T>::AppendContactKinematicsMpm(
    const systems::Context<T>& context,
    std::vector<ContactPairKinematics<T>>* result) const {

    const MultibodyTreeTopology& tree_topology = manager_->internal_tree().get_topology();
    const TreeIndex clique_index_mpm(tree_topology.num_trees() + num_deformable_bodies() + 1);

    /* Retrieve (or compute) all mpm contact, and compute Jmpm */
    MpmContact<T> mpm_contact;
    CalcMpmContact(context, &mpm_contact); // TBD: make this an eval

    const MpmState<T>& current_mpm_state = EvalMpmState(context);
    MpmSolverScratchData<T>& scratch =
    manager_->plant()
        .get_cache_entry(cache_indexes_.mpm_solver_scratch)
        .get_mutable_cache_entry_value(context)
        .template GetMutableValueOrThrow<MpmSolverScratchData<T>>();
    std::unordered_map<int, MatrixX<T>> map_contact_particle_to_Jacobian;
    mpm_solver_->ComputeJacobianMpm(current_mpm_state, mpm_contact,
                     scratch, &map_contact_particle_to_Jacobian); 
    // if a particle is a contact particle, this will give Jmpm s.t. vc_mpm_W = Jmpm * v_mpm. Jmpm has dimension 3 by num_grid_nodes*3

    constexpr int kZAxis = 2;
    const int nv = manager_->plant().num_velocities();


    for (size_t i = 0; i < mpm_contact.GetNumContactPairs(); ++i) {
        // for each contact pair, want J = R_CW * [-Jmpm | Jrigid]

        /* Compute the rotation matrix R_CW */
        math::RotationMatrix<T> R_WC = math::RotationMatrix<T>::MakeFromOneUnitVector(mpm_contact.GetNormalAt(i), kZAxis);
        const math::RotationMatrix<T> R_CW = R_WC.transpose();

        /* We have at most two blocks per contact. */
        std::vector<typename ContactPairKinematics<T>::JacobianTreeBlock> jacobian_blocks;
        jacobian_blocks.reserve(2);

        /* MPM part of Jacobian */
        MatrixX<T> J_mpm = map_contact_particle_to_Jacobian.at(mpm_contact.GetParticleIndexAt(i));
        jacobian_blocks.emplace_back(clique_index_mpm, MatrixBlock<T>(std::move(J_mpm)));

        /* Non-MPM (rigid) part of Jacobian */
        const BodyIndex index_B = manager_->geometry_id_to_body_index().at(mpm_contact.GetNonMpmIdAt(i));
        const TreeIndex tree_index_rigid = tree_topology.body_to_tree_index(index_B);
        if (tree_index_rigid.is_valid()) {
            Matrix3X<T> Jv_v_WBc_W(3, nv);
            const Body<T>& rigid_body = manager_->plant().get_body(index_B);
            const Frame<T>& frame_W = manager_->plant().world_frame();
            manager_->internal_tree().CalcJacobianTranslationalVelocity(
                context, JacobianWrtVariable::kV, rigid_body.body_frame(), frame_W,
                mpm_contact.GetContactPositionAt(i), frame_W, frame_W, &Jv_v_WBc_W);
            Matrix3X<T> J_rigid = R_CW.matrix() * Jv_v_WBc_W.middleCols(
                                    tree_topology.tree_velocities_start(tree_index_rigid),
                                    tree_topology.num_tree_velocities(tree_index_rigid));
            jacobian_blocks.emplace_back(tree_index_rigid, MatrixBlock<T>(std::move(J_rigid)));
        }

        // configuration part
        const int objectA = tree_topology.num_trees() + num_deformable_bodies() + 1; // not really used
        const int objectB = index_B; // not really used
        const Vector3<T> p_ApC_W{0,0,0}; // TODO: change this. it will be used in SapFrictionConeConstraint<T>::DoAccumulateSpatialImpulses
        // Contact point position relative to rigid body B, same as in FEM-rigid
        const math::RigidTransform<T>& X_WB = manager_->plant().EvalBodyPoseInWorld(context, manager_->plant().get_body(index_B));
        const Vector3<T>& p_WB = X_WB.translation();
        const Vector3<T> p_BC_W = mpm_contact.GetContactPositionAt(i) - p_WB;

        ContactConfiguration<T> configuration{
          .objectA = objectA, // not really used
          .p_ApC_W = p_ApC_W, // to be changed, see above
          .objectB = objectB, // not really used
          .p_BqC_W = p_BC_W,
          .phi = mpm_contact.GetPenetrationDistanceAt(i),
          .R_WC = R_WC};

        result->emplace_back(std::move(jacobian_blocks), std::move(configuration));
    }


    




    
    


}


template <typename T>
void DeformableDriver<T>::AppendContactKinematics(
    const systems::Context<T>& context,
    std::vector<ContactPairKinematics<T>>* result) const {
  DRAKE_DEMAND(result != nullptr);
  /* 
  
  
  
  Sap needs Jc such that vc_C = Jc * v where v is the generalized velocities for participating dofs.
    `vc_C` is defined with C frame being the contact frame, centered at the contact point,
  with Cz parallel to the contact normal.  vc_C = R_CW * vc_W. We can get R_CW from the contact normal.
  Need vc_W.
  vc_W = v_ApBq_W = v_WBq - v_WAp where Bq is point on B incident to the contact point,
  and Ap is the point on A incident to the contact point.
  CalcJacobianTranslationalVelocity gives Jb such that v_WBq = Jb * vb where vb is the generalized velocity
  for rigid dofs.
  
  
  
  Since v_AcBc_W = v_WBc - v_WAc the relative velocity Jacobian will be:
     Jv_v_AcBc_W = Jv_v_WBc_W - Jv_v_WAc_W.
   That is the relative velocity at C is v_AcBc_W = Jv_v_AcBc_W * v.
   Finally Jv_v_AcBc_C = R_WC.transpose() * Jv_v_AcBc_W.
   Currently, only deformable vs rigid contact is supported. Deformable
   body is body A and rigid body is body B. Moreover, the set of dofs for
   deformable bodies and rigid bodies are mutually exclusive, and
   Jv_v_WAc_W = 0 for rigid dofs and Jv_v_WBc_W = 0 for deformable dofs. As a
   result, we know the size of Jv_v_WBc_W up front. */
  const int nv = manager_->plant().num_velocities();
  Matrix3X<T> Jv_v_WBc_W(3, nv);
  const MultibodyTreeTopology& tree_topology =
      manager_->internal_tree().get_topology();
  const DeformableContact<T>& deformable_contact =
      EvalDeformableContact(context);
  const std::vector<DeformableContactSurface<T>>& contact_surfaces =
      deformable_contact.contact_surfaces();
  for (const auto& surface : contact_surfaces) {
    const GeometryId id_A = surface.id_A();
    const GeometryId id_B = surface.id_B();
    /* Body A is guaranteed to be deformable. */
    const DeformableBodyIndex index_A =
        deformable_model_->GetBodyIndex(deformable_model_->GetBodyId(id_A));
    const TreeIndex clique_index_A(tree_topology.num_trees() + index_A);
    const ContactParticipation& participation =
        deformable_contact.contact_participation(id_A);
    const PartialPermutation& vertex_permutation =
        EvalVertexPermutation(context, id_A);
    /* For now, body B is guaranteed to be rigid. */
    DRAKE_DEMAND(!surface.is_B_deformable());
    /* Retrieve the boundary condition information of body A to determine
     which columns for the jacobian need to be zero out later. */
    const DeformableBodyId body_id_A = deformable_model_->GetBodyId(id_A);
    const FemModel<T>& fem_model = deformable_model_->GetFemModel(body_id_A);
    const DirichletBoundaryCondition<T>& bc =
        fem_model.dirichlet_boundary_condition();
    /* The number of boundary conditions added to each vertex. */
    std::vector<int> num_bcs(fem_model.num_nodes(), 0);
    for (const auto& it : bc.index_to_boundary_state()) {
      /* Note that we currently only allow zero boundary conditions. */
      DRAKE_DEMAND(it.second.v == Vector3<T>::Zero());
      DRAKE_DEMAND(it.second.a == Vector3<T>::Zero());
      const int vertex_index = it.first;
      ++num_bcs[vertex_index];
    }

    for (int i = 0; i < surface.num_contact_points(); ++i) {
      /* We have at most two blocks per contact. */
      std::vector<typename ContactPairKinematics<T>::JacobianTreeBlock>
          jacobian_blocks;
      jacobian_blocks.reserve(2);
      /* Contact solver assumes the normal points from A to B whereas the
       surface's normal points from B to A. */
      const Vector3<T>& nhat_W = -surface.nhats_W()[i];
      constexpr int kZAxis = 2;
      math::RotationMatrix<T> R_WC =
          math::RotationMatrix<T>::MakeFromOneUnitVector(nhat_W, kZAxis);
      const math::RotationMatrix<T> R_CW = R_WC.transpose();
      /* Calculate the jacobian block for the body A. */
      Block3x3SparseMatrix<T> negative_Jv_v_WAc_C(
          1, participation.num_vertices_in_contact());
      Vector4<int> participating_vertices =
          surface.contact_vertex_indexes_A()[i];
      const Vector4<T>& b = surface.barycentric_coordinates_A()[i];
      std::vector<typename Block3x3SparseMatrix<T>::Triplet> triplets;
      triplets.reserve(4);
      for (int v = 0; v < 4; ++v) {
        const bool vertex_under_bc = num_bcs[participating_vertices(v)] > 0;
        /* Map indexes to the permuted domain. */
        participating_vertices(v) =
            vertex_permutation.permuted_index(participating_vertices(v));
        if (!vertex_under_bc) {
          /* v_WAc = (b₀ * v₀ + b₁ * v₁ + b₂ * v₂ + b₃ * v₃) where v₀, v₁, v₂,
           v₃ are the velocities of the vertices forming the tetrahedron
           containing the contact point and the b's are their corresponding
           barycentric weights. */
          triplets.emplace_back(0, participating_vertices(v),
                                -b(v) * R_CW.matrix());
        }
        negative_Jv_v_WAc_C.SetFromTriplets(triplets);
        /* If the vertex is under bc, the corresponding jacobain block is zero
         because the vertex doesn't contribute to the contact velocity. */
      }
      jacobian_blocks.emplace_back(
          clique_index_A, MatrixBlock<T>(std::move(negative_Jv_v_WAc_C)));

      /* Calculate the jacobian block for the rigid body B if it's not static.
       */
      const BodyIndex index_B = manager_->geometry_id_to_body_index().at(id_B);
      const TreeIndex tree_index = tree_topology.body_to_tree_index(index_B);
      const Vector3<T>& p_WC = surface.contact_points_W()[i];
      if (tree_index.is_valid()) {
        const Body<T>& rigid_body = manager_->plant().get_body(index_B);
        const Frame<T>& frame_W = manager_->plant().world_frame();
        manager_->internal_tree().CalcJacobianTranslationalVelocity(
            context, JacobianWrtVariable::kV, rigid_body.body_frame(), frame_W,
            p_WC, frame_W, frame_W, &Jv_v_WBc_W);
        Matrix3X<T> J =
            R_CW.matrix() * Jv_v_WBc_W.middleCols(
                                tree_topology.tree_velocities_start(tree_index),
                                tree_topology.num_tree_velocities(tree_index));
        jacobian_blocks.emplace_back(tree_index, MatrixBlock<T>(std::move(J)));
      }

      // Contact configuration between objects A and B.
      // By convention, deformable bodies are assigned object indexes after all
      // rigid bodies.
      const int objectA =
          index_A + manager_->plant().num_bodies();  // Deformable body.
      const int objectB = index_B;                   // Rigid body.

      // Contact point position relative to deformable object A. For deformable
      // objects, we'll use the centroid of the contact surface as the
      // relative-to point in the configuration.
      const Vector3<T>& p_WCentroid = surface.contact_mesh_W().centroid();
      const Vector3<T> p_ACentroidC_W = p_WC - p_WCentroid;

      // Contact point position relative to rigid body B.
      const math::RigidTransform<T>& X_WB =
          manager_->plant().EvalBodyPoseInWorld(
              context, manager_->plant().get_body(index_B));
      const Vector3<T>& p_WB = X_WB.translation();
      const Vector3<T> p_BC_W = p_WC - p_WB;

      ContactConfiguration<T> configuration{
          .objectA = objectA,
          .p_ApC_W = p_ACentroidC_W,
          .objectB = objectB,
          .p_BqC_W = p_BC_W,
          .phi = surface.signed_distances()[i],
          .R_WC = R_WC};
      result->emplace_back(std::move(jacobian_blocks),
                           std::move(configuration));
    }
  }
}

template <typename T>
void DeformableDriver<T>::CalcDiscreteStates(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* next_states) const {
  const int num_bodies = deformable_model_->num_bodies();
  if (num_bodies > 0){
    for (DeformableBodyIndex index(0); index < num_bodies; ++index) {
        const FemState<T>& next_fem_state = EvalNextFemState(context, index);
        const int num_dofs = next_fem_state.num_dofs();
        // Update the discrete values.
        VectorX<T> discrete_value(num_dofs * 3);
        discrete_value.head(num_dofs) = next_fem_state.GetPositions();
        discrete_value.segment(num_dofs, num_dofs) = next_fem_state.GetVelocities();
        discrete_value.tail(num_dofs) = next_fem_state.GetAccelerations();
        const DeformableBodyId id = deformable_model_->GetBodyId(index);
        next_states->set_value(deformable_model_->GetDiscreteStateIndex(id),
                            discrete_value);
    }
  }
  
}

template <typename T>
void DeformableDriver<T>::CalcAbstractStates(
    const systems::Context<T>& context,
    systems::State<T>* update) const {
  
  if (deformable_model_->ExistsMpmModel()) {

    const MpmState<T>& next_mpm_state = EvalNextMpmState(context);

    // the two lines below together act as setting abstract state
    MpmState<T>& mutable_mpm_state = update->template get_mutable_abstract_state<MpmState<T>>(deformable_model_->GetParticlesAbstractIndex());
    mutable_mpm_state.SetState(next_mpm_state);
    
  }
}

template <typename T>
Multiplexer<T>::Multiplexer(std::vector<int> sizes) : sizes_(std::move(sizes)) {
  DRAKE_THROW_UNLESS(!sizes_.empty());
  DRAKE_THROW_UNLESS(sizes_[0] >= 0);
  offsets_.resize(num_vectors());
  offsets_[0] = 0;
  for (int i = 1; i < num_vectors(); ++i) {
    DRAKE_THROW_UNLESS(sizes_[i] >= 0);
    offsets_[i] = offsets_[i - 1] + sizes_[i - 1];
  }
  num_entries_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);
}

template <typename T>
VectorX<T> Multiplexer<T>::Multiplex(std::vector<VectorX<T>>&& inputs) const {
  VectorX<T> result(num_entries_);
  DRAKE_THROW_UNLESS(static_cast<int>(inputs.size()) == num_vectors());
  for (int i = 0; i < num_vectors(); ++i) {
    DRAKE_THROW_UNLESS(sizes_[i] == inputs[i].size());
    /* We move inputs[i] into the block vector so that (hopefully) the move
     semantic is applied to individual entries. This may be significant when
     T != double. */
    result.segment(offsets_[i], sizes_[i]) = std::move(inputs[i]);
  }
  return result;
}

template <typename T>
Eigen::Ref<const VectorX<T>> Multiplexer<T>::Demultiplex(
    const Eigen::Ref<const VectorX<T>>& input, int index) const {
  DRAKE_THROW_UNLESS(0 <= index && index < num_vectors());
  DRAKE_THROW_UNLESS(input.size() == num_entries_);
  /* We prefer Eigen::Ref<const VectorX<T>> to Eigen::VectorBlock<const
   Eigen::Ref<const VectorX<T>>> for readability. */
  return input.segment(offsets_[index], sizes_[index]);
}

template <typename T>
Eigen::Ref<VectorX<T>> Multiplexer<T>::Demultiplex(EigenPtr<VectorX<T>> input,
                                                   int index) const {
  DRAKE_THROW_UNLESS(0 <= index && index < num_vectors());
  DRAKE_THROW_UNLESS(input->size() == num_entries_);
  /* We prefer Eigen::Ref<VectorX<T>> to
   Eigen::VectorBlock<Eigen::Ref<VectorX<T>>> for readability and encapsulation
   of the implementation of EigenPtr. */
  return input->segment(offsets_[index], sizes_[index]);
}

template <typename T>
void DeformableDriver<T>::CalcFemState(const Context<T>& context,
                                       DeformableBodyIndex index,
                                       FemState<T>* fem_state) const {
  const DeformableBodyId id = deformable_model_->GetBodyId(index);
  const systems::BasicVector<T>& discrete_state =
      context.get_discrete_state().get_vector(
          deformable_model_->GetDiscreteStateIndex(id));
  const VectorX<T>& discrete_value = discrete_state.value();
  DRAKE_DEMAND(discrete_value.size() % 3 == 0);
  const int num_dofs = discrete_value.size() / 3;
  const auto& q = discrete_value.head(num_dofs);
  const auto& qdot = discrete_value.segment(num_dofs, num_dofs);
  const auto& qddot = discrete_value.tail(num_dofs);
  fem_state->SetPositions(q);
  fem_state->SetVelocities(qdot);
  fem_state->SetAccelerations(qddot);
}

template <typename T>
void DeformableDriver<T>::CalcMpmState(const Context<T>& context,
                                       MpmState<T>* mpm_state) const {

    const MpmState<T>& current_state = context.template get_abstract_state<MpmState<T>>(deformable_model_->particles_container_index());
    mpm_state->SetState(current_state);
}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalFemState(
    const Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
const MpmState<T>& DeformableDriver<T>::EvalMpmState(
    const Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.mpm_state)
      .template Eval<MpmState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionFemState(
    const systems::Context<T>& context, DeformableBodyIndex index,
    FemState<T>* fem_state_star) const {
  const FemState<T>& fem_state = EvalFemState(context, index);
  const DeformableBodyId id = deformable_model_->GetBodyId(index);
  const FemModel<T>& model = deformable_model_->GetFemModel(id);
  // TODO(xuchenhan-tri): We should expose an API to set the solver tolerance
  // here.
  const FemSolver<T> solver(&model, integrator_.get());
  FemSolverScratchData<T>& scratch =
      manager_->plant()
          .get_cache_entry(cache_indexes_.fem_solver_scratches.at(index))
          .get_mutable_cache_entry_value(context)
          .template GetMutableValueOrThrow<FemSolverScratchData<T>>();
  solver.AdvanceOneTimeStep(fem_state, fem_state_star, &scratch);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionMpmState(
    const systems::Context<T>& context,MpmState<T>* mpm_state_star) const {

  const MpmState<T>& mpm_state = EvalMpmState(context);

  MpmSolverScratchData<T>& scratch =
     manager_->plant()
         .get_cache_entry(cache_indexes_.mpm_solver_scratch)
         .get_mutable_cache_entry_value(context)
         .template GetMutableValueOrThrow<MpmSolverScratchData<T>>();
  mpm_solver_->AdvanceOneTimeStep(mpm_state, mpm_state_star, &scratch);

  // mpm_solver_->AdvanceOneTimeStep(mpm_state, mpm_state_star);
  std::cout << "finish CalcFreeMotionMpmState" << std::endl;

}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalFreeMotionFemState(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.free_motion_fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
const MpmState<T>& DeformableDriver<T>::EvalFreeMotionMpmState(
    const systems::Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.free_motion_mpm_state)
      .template Eval<MpmState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcNextFemState(const systems::Context<T>& context,
                                           DeformableBodyIndex index,
                                           FemState<T>* next_fem_state) const {
  const DeformableContact<T>& contact_data = EvalDeformableContact(context);
  const GeometryId g_id =
      deformable_model_->GetGeometryId(deformable_model_->GetBodyId(index));
  const ContactParticipation& participation =
      contact_data.contact_participation(g_id);
  if (participation.num_vertices_in_contact() == 0) {
    // TODO(xuchenhan-tri): Account for constraints when we support them.
    /* The next states are the free motion states if no vertex of the
     deformable body participates in contact. */
    const FemState<T>& free_motion_state =
        EvalFreeMotionFemState(context, index);
    next_fem_state->SetPositions(free_motion_state.GetPositions());
    next_fem_state->SetVelocities(free_motion_state.GetVelocities());
    next_fem_state->SetAccelerations(free_motion_state.GetAccelerations());
  } else {
    const ContactSolverResults<T>& results =
        manager_->EvalContactSolverResults(context);
    /* Get the next time step velocities and free motion velocities for *all*
     participating deformable dofs (belonging to *all* deformable bodies). */
    const int num_participating_dofs =
        results.v_next.size() - manager_->plant().num_velocities();
    const auto participating_v_next =
        results.v_next.tail(num_participating_dofs);
    const VectorX<T>& participating_v_star =
        EvalParticipatingFreeMotionVelocities(context);
    /* The next time step velocities and free motion velocities for *this*
     body. */
    const Multiplexer<T>& mux = EvalParticipatingVelocityMultiplexer(context);
    const Eigen::Ref<const VectorX<T>> body_participating_v_star =
        mux.Demultiplex(participating_v_star, index);
    const Eigen::Ref<const VectorX<T>> body_participating_v_next =
        mux.Demultiplex(participating_v_next, index);
    const VectorX<T> body_participating_dv =
        body_participating_v_next - body_participating_v_star;
    /* Compute the value of the post-constraint non-participating
     velocities using Schur complement. */
    const SchurComplement<T>& schur_complement =
        EvalFreeMotionTangentMatrixSchurComplement(context, index);
    const VectorX<T> body_nonparticipating_dv =
        schur_complement.SolveForY(body_participating_dv);
    /* Concatenate the participating and non-participating velocities and
     then apply the inverse permutation to put the dofs in their original
     order. */
    const PartialPermutation& full_velocity_permutation =
        participation.CalcDofPermutation();
    const int num_dofs = full_velocity_permutation.domain_size();
    VectorX<T> permuted_dv(num_dofs);
    permuted_dv << body_participating_dv, body_nonparticipating_dv;
    VectorX<T> dv(num_dofs);
    full_velocity_permutation.ApplyInverse(permuted_dv, &dv);
    /* v_next = v_star + dv */
    VectorX<T>& v_next = dv;
    v_next += EvalFreeMotionFemState(context, index).GetVelocities();
    const FemState<T>& fem_state = EvalFemState(context, index);
    integrator_->AdvanceOneTimeStep(fem_state, v_next, next_fem_state);
  }
}


template <typename T>
void DeformableDriver<T>::CalcNextMpmState(const systems::Context<T>& context,
                                           MpmState<T>* next_mpm_state) const {
  
    const MpmState<T>& free_motion_state = EvalFreeMotionMpmState(context);
    next_mpm_state->SetParticles(free_motion_state.GetParticles());
}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalNextFemState(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.next_fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
const MpmState<T>& DeformableDriver<T>::EvalNextMpmState(
    const systems::Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.next_mpm_state)
      .template Eval<MpmState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcDeformableContact(
    const Context<T>& context, DeformableContact<T>* result) const {
  const geometry::QueryObject<T>& query_object =
      manager_->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  query_object.ComputeDeformableContact(result);
}

template <typename T>
void DeformableDriver<T>::CalcMpmContact(
    const Context<T>& context, MpmContact<T>* result) const {

    result->Reset();
    const geometry::QueryObject<T>& query_object = manager_->plant().get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
    const MpmState<T>& current_mpm_state = EvalMpmState(context);
    const Particles<T>& current_particles = current_mpm_state.GetParticles(); 
    for (int i = 0; i < current_particles.get_num_particles(); i++) {
        std::vector<geometry::SignedDistanceToPoint<T>> point_to_geometry = query_object.ComputeSignedDistanceToPoint(current_particles.get_position(i));
        for (size_t num_geometries = 0; num_geometries < point_to_geometry.size(); num_geometries++) {
            const T signed_distance = point_to_geometry[num_geometries].distance;
            if (signed_distance < 0) {
                // if mpm particle is inside rigid geometry, i.e. in contact
                const GeometryId& id_rigid = point_to_geometry[num_geometries].id_G;
                const Vector3<T>& normal = point_to_geometry[num_geometries].grad_W; // normal direction ( <---- p      )
                result->AddMpmContactPair(i, id_rigid, signed_distance, normal, current_particles.get_position(i));
               
            }
        }
    }

}

template <typename T>
const DeformableContact<T>& DeformableDriver<T>::EvalDeformableContact(
    const Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.deformable_contact)
      .template Eval<DeformableContact<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcDofPermutation(const Context<T>& context,
                                             DeformableBodyIndex index,
                                             PartialPermutation* result) const {
  const DeformableBodyId body_id = deformable_model_->GetBodyId(index);
  const GeometryId geometry_id = deformable_model_->GetGeometryId(body_id);
  *result = EvalDeformableContact(context)
                .contact_participation(geometry_id)
                .CalcDofPartialPermutation();
}

template <typename T>
const PartialPermutation& DeformableDriver<T>::EvalDofPermutation(
    const Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.dof_permutations.at(index))
      .template Eval<PartialPermutation>(context);
}

template <typename T>
void DeformableDriver<T>::CalcVertexPermutation(
    const Context<T>& context, GeometryId id,
    PartialPermutation* result) const {
  *result = EvalDeformableContact(context)
                .contact_participation(id)
                .CalcVertexPartialPermutation();
}

template <typename T>
const PartialPermutation& DeformableDriver<T>::EvalVertexPermutation(
    const Context<T>& context, GeometryId id) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.vertex_permutations.at(id))
      .template Eval<PartialPermutation>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingVelocityMultiplexer(
    const Context<T>& context, Multiplexer<T>* result) const {
  const int num_bodies = deformable_model_->num_bodies();
  std::vector<int> num_participating_dofs(num_bodies);
  for (DeformableBodyIndex i(0); i < num_bodies; ++i) {
    num_participating_dofs[i] =
        EvalDofPermutation(context, i).permuted_domain_size();
  }
  *result = Multiplexer<T>(std::move(num_participating_dofs));
}

template <typename T>
const Multiplexer<T>& DeformableDriver<T>::EvalParticipatingVelocityMultiplexer(
    const Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.participating_velocity_mux)
      .template Eval<Multiplexer<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingVelocities(
    const Context<T>& context, VectorX<T>* result) const {
  const int num_bodies = deformable_model_->num_bodies();
  std::vector<VectorX<T>> participating_velocities(num_bodies);
  for (DeformableBodyIndex i(0); i < num_bodies; ++i) {
    const PartialPermutation& permutation = EvalDofPermutation(context, i);
    const VectorX<T>& v = EvalFemState(context, i).GetVelocities();
    participating_velocities[i].resize(permutation.permuted_domain_size());
    permutation.Apply(v, &participating_velocities[i]);
  }
  *result = EvalParticipatingVelocityMultiplexer(context).Multiplex(
      std::move(participating_velocities));
}

template <typename T>
const VectorX<T>& DeformableDriver<T>::EvalParticipatingVelocities(
    const Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.participating_velocities)
      .template Eval<VectorX<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingFreeMotionVelocities(
    const Context<T>& context, VectorX<T>* result) const {
  DRAKE_DEMAND(result != nullptr);
  const int num_bodies = deformable_model_->num_bodies();
  std::vector<VectorX<T>> participating_v_star(num_bodies);
  for (DeformableBodyIndex i(0); i < num_bodies; ++i) {
    const PartialPermutation& permutation = EvalDofPermutation(context, i);
    const VectorX<T>& v_star =
        EvalFreeMotionFemState(context, i).GetVelocities();
    participating_v_star[i].resize(permutation.permuted_domain_size());
    permutation.Apply(v_star, &participating_v_star[i]);
  }
  if (num_bodies>0) {
  // TODO(xuchenhan-tri): Consider adding a in-place version of Multiplex.
  *result = EvalParticipatingVelocityMultiplexer(context).Multiplex(
      std::move(participating_v_star));}
}

template <typename T>
const VectorX<T>& DeformableDriver<T>::EvalParticipatingFreeMotionVelocities(
    const Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.participating_free_motion_velocities)
      .template Eval<VectorX<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionTangentMatrix(
    const systems::Context<T>& context, DeformableBodyIndex index,
    PetscSymmetricBlockSparseMatrix* tangent_matrix) const {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  const DeformableBodyId id = deformable_model_->GetBodyId(index);
  const FemModel<T>& fem_model = deformable_model_->GetFemModel(id);
  const FemState<T>& fem_state_star = EvalFreeMotionFemState(context, index);
  /* Multiply by dt because we need the tangent matrix of the momentum balance
   instead of the force balance. */
  fem_model.CalcTangentMatrix(
      fem_state_star, manager_->plant().time_step() * integrator_->GetWeights(),
      tangent_matrix);
  tangent_matrix->AssembleIfNecessary();
}

template <typename T>
const PetscSymmetricBlockSparseMatrix&
DeformableDriver<T>::EvalFreeMotionTangentMatrix(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.free_motion_tangent_matrices.at(index))
      .template Eval<PetscSymmetricBlockSparseMatrix>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionTangentMatrixSchurComplement(
    const systems::Context<T>& context, DeformableBodyIndex index,
    SchurComplement<T>* result) const {
  DRAKE_DEMAND(result != nullptr);
  const DeformableContact<T>& contact_data = EvalDeformableContact(context);
  const DeformableBodyId body_id = deformable_model_->GetBodyId(index);
  const GeometryId geometry_id = deformable_model_->GetGeometryId(body_id);
  /* Avoid the expensive tangent matrix and Schur complement calculation if
   there's no contact at all. */
  const ContactParticipation& body_contact_participation =
      contact_data.contact_participation(geometry_id);
  if (body_contact_participation.num_vertices_in_contact() == 0) {
    *result = SchurComplement<T>();
    return;
  }
  const PetscSymmetricBlockSparseMatrix& tangent_matrix =
      EvalFreeMotionTangentMatrix(context, index);
  std::vector<int> participating_vertices;
  std::vector<int> non_participating_vertices;
  const PartialPermutation& permutation =
      EvalVertexPermutation(context, geometry_id);
  DRAKE_DEMAND(3 * permutation.domain_size() == tangent_matrix.cols());
  for (int v = 0; v < permutation.domain_size(); ++v) {
    if (permutation.participates(v)) {
      participating_vertices.emplace_back(v);
    } else {
      non_participating_vertices.emplace_back(v);
    }
  }
  *result = tangent_matrix.CalcSchurComplement(non_participating_vertices,
                                               participating_vertices);
}

template <typename T>
const SchurComplement<T>&
DeformableDriver<T>::EvalFreeMotionTangentMatrixSchurComplement(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(
          cache_indexes_.free_motion_tangent_matrix_schur_complements.at(index))
      .template Eval<SchurComplement<T>>(context);
}

template class DeformableDriver<double>;
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::internal::Multiplexer);

}  // namespace internal
}  // namespace multibody
}  // namespace drake
