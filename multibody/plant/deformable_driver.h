#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/query_results/mpm_particle_contact_pair.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/contact_solvers/sap/sap_fixed_constraint.h"
#include "drake/multibody/contact_solvers/schur_complement.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/fem/fem_solver.h"
#include "drake/multibody/mpm/conjugate_gradient.h"
#include "drake/multibody/mpm/mpm_solver.h"
#include "drake/multibody/mpm/mpm_state.h"
#include "drake/multibody/mpm/mpm_transfer.h"
#include "drake/multibody/mpm/particles_to_bgeo.h"
#include "drake/multibody/plant/contact_pair_kinematics.h"
#include "drake/multibody/plant/deformable_contact_info.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/discrete_contact_data.h"
#include "drake/multibody/plant/discrete_contact_pair.h"
#include "drake/systems/framework/context.h"

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

template <typename T>
class DiscreteUpdateManager;

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

  // ------------ newly added for MPM ---------------
  bool SapIncludesMpm() const { return false; }

  bool ExistsMpmBody() const { return deformable_model_->ExistsMpmModel(); }

  void CalcAbstractStatesExplicit(const systems::Context<T>& context,
                                  systems::State<T>* update) const {
    if (deformable_model_->ExistsMpmModel()) {
      // WriteMpmParticlesToBgeo(context);

      const mpm::MpmPostContactResult<T>& result =
          EvalMpmPostContactResult(context);
      mpm::MpmState<T>& mutable_mpm_state =
          update->template get_mutable_abstract_state<mpm::MpmState<T>>(
              deformable_model_->mpm_model().mpm_state_index());
      mutable_mpm_state = result.mpm_state;
    }
    std::cout << "time: " << context.get_time() << std::endl;
  }

  const mpm::MpmPostContactResult<T>& EvalMpmPostContactResult(
      const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.mpm_post_contact_result)
        .template Eval<mpm::MpmPostContactResult<T>>(context);
  }

  void CalcMpmPostContactResult(const systems::Context<T>& context,
                                mpm::MpmPostContactResult<T>* result) const {
    result->forces_to_rigid_bodies.clear();
    result->mobod_indices.clear();
    // first duplicate currenet state
    result->mpm_state = context.template get_abstract_state<mpm::MpmState<T>>(
        deformable_model_->mpm_model().mpm_state_index());

    mpm::MpmSolverScratch<T>& mpm_scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.mpm_solver_scratch)
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();

    double large_dt = manager_->plant().time_step();
    int num_mpm_steps = deformable_model_->maniskill_params.num_mpm_substeps;
    double mpm_dt = large_dt / num_mpm_steps;

    double contact_stiffness =
        deformable_model_->maniskill_params.contact_stiffness;
    double contact_damping =
        deformable_model_->maniskill_params.contact_damping;
    double kf = deformable_model_->maniskill_params.friction_kf;
    double friction_mu = deformable_model_->maniskill_params.friction_mu;
    Vector3<T> f_total_left(0, 0, 0);
    Vector3<T> f_total_right(0, 0, 0);
    Vector3<T> v_total_left(0, 0, 0);
    Vector3<T> v_total_right(0, 0, 0);
    double time = context.get_time();
    int current_step = std::round(time / manager_->plant().time_step());
    int ratio = std::round(0.005 / manager_->plant().time_step());
    ratio = 625;
    unused(f_total_left, f_total_right, v_total_left, v_total_right);
    int left_count = 0;
    int right_count = 0;

    for (int step = 0; step < num_mpm_steps; ++step) {
      // first compute contact pairs
      std::vector<geometry::internal::MpmParticleContactPair<T>>
          mpm_contact_pairs;
      CalcMpmContactPairsWithParticlesInput(
          context, result->mpm_state.particles, &mpm_contact_pairs);
      // next, for each contact pair, compute and apply contact force
      for (size_t i = 0; i < mpm_contact_pairs.size(); ++i) {
        // std::cout << "contact point x " <<
        // mpm_contact_pairs[i].particle_in_contact_position(0) << std::endl;
        // std::cout << "rigid index: " << mpm_contact_pairs[i].non_mpm_id <<
        // std::endl;
        const Vector3<T>& contact_normal = mpm_contact_pairs[i].normal;
        size_t p = mpm_contact_pairs[i].particle_in_contact_index;
        double penetration_distance = mpm_contact_pairs[i].penetration_distance;
        // get relative velocity in contact frame
        constexpr int kZAxis = 2;
        math::RotationMatrix<T> R_WC =
            math::RotationMatrix<T>::MakeFromOneUnitVector(contact_normal,
                                                           kZAxis);
        const math::RotationMatrix<T> R_CW = R_WC.transpose();

        Vector3<T> relative_v_C =
            R_CW.matrix() * result->mpm_state.particles.GetVelocityAt(p);
        const MultibodyTreeTopology& tree_topology =
            manager_->internal_tree().get_topology();
        const int nv = manager_->plant().num_velocities();
        const BodyIndex index_B = manager_->geometry_id_to_body_index().at(
            mpm_contact_pairs[i].non_mpm_id);
        const TreeIndex tree_index_rigid =
            tree_topology.body_to_tree_index(index_B);
        if (tree_index_rigid.is_valid()) {
          Matrix3X<T> Jv_v_WBc_W(3, nv);
          const Body<T>& rigid_body = manager_->plant().get_body(index_B);
          const Frame<T>& frame_W = manager_->plant().world_frame();
          manager_->internal_tree().CalcJacobianTranslationalVelocity(
              context, JacobianWrtVariable::kV, rigid_body.body_frame(),
              frame_W, mpm_contact_pairs[i].particle_in_contact_position,
              frame_W, frame_W, &Jv_v_WBc_W);
          Matrix3X<T> J_rigid =
              R_CW.matrix() *
              Jv_v_WBc_W.middleCols(
                  tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                  tree_topology.num_tree_velocities(tree_index_rigid));
          const Eigen::VectorBlock<const VectorX<T>> v =
              manager_->plant().GetVelocities(context);
          relative_v_C -=
              J_rigid *
              v.segment(
                  tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                  tree_topology.num_tree_velocities(tree_index_rigid));
        }
        // this normal points toward more penetration
        double fn = std::abs(penetration_distance) *
                    contact_stiffness;  // acting towards the reverse direction
                                        // of normal
        double fd = contact_damping *
                    std::min(relative_v_C(2), 0.0);  // this will be negative

        Vector2<T> vt(relative_v_C(0), relative_v_C(1));
        Vector2<T> ft = vt.normalized() * std::min(kf*vt.norm(), friction_mu * fn);
        // double ft0 = std::min(std::abs(relative_v_C(0) * kf), friction_mu * fn);
        // if (relative_v_C(0) < 0) {
        //   ft0 = -ft0;
        // }
        // double ft1 = std::min(std::abs(relative_v_C(1) * kf), friction_mu * fn);
        // if (relative_v_C(1) < 0) {
        //   ft1 = -ft1;
        // }
        Vector3<T> contact_force_C(-ft(0), -ft(1), -(fn + fd));
        Vector3<T> contact_force_W = R_WC * contact_force_C;
        // apply contact force to mpm
        const Vector3<T>& current_vp =
            result->mpm_state.particles.GetVelocityAt(p);
        result->mpm_state.particles.SetVelocityAt(
            p, current_vp + contact_force_W * mpm_dt /
                                result->mpm_state.particles.GetMassAt(p));

        // also store contact force for rigid body
        const BodyIndex rigid_body_index =
            manager_->geometry_id_to_body_index().at(
                mpm_contact_pairs[i].non_mpm_id);
        const auto& rigid_body = manager_->plant().get_body(rigid_body_index);
        const auto body_mobod_index = rigid_body.mobod_index();
        SpatialForce<T> spatial_force_C_W(Vector3<T>::Zero(),
                                          -contact_force_W / num_mpm_steps);
        result->mobod_indices.push_back(body_mobod_index);
        const Vector3<T>& Bo = manager_->plant()
                                   .EvalBodyPoseInWorld(context, rigid_body)
                                   .translation();
        result->forces_to_rigid_bodies.emplace_back(spatial_force_C_W.Shift(
            Bo - mpm_contact_pairs[i].particle_in_contact_position));

        if ((step == 0) && (current_step % ratio) == 0) {
          if ((mpm_contact_pairs[i].non_mpm_id.get_value()) == 21) {

            if (mpm_contact_pairs[i].particle_in_contact_position(0) < 0) {
              f_total_left += contact_force_W;
              v_total_left += R_WC * relative_v_C;
              left_count++;
            } else {
              f_total_right += contact_force_W;
              v_total_right += R_WC * relative_v_C;
              right_count++;
            }
          }
        }
      }

      if ((step == 0) && (current_step % ratio) == 0) {
        if (left_count > 0) {
            v_total_left = v_total_left / left_count;
        }
        if (right_count > 0) {
            v_total_right = v_total_right / right_count;
        }
        
        if (time == 0.0) {
          std::ofstream F("output.txt");
          F << f_total_left(0) << ", " << f_total_left(1) << ", "
            << f_total_left(2) << "," << f_total_right(0) << ", "
            << f_total_right(1) << ", " << f_total_right(2) << ","
            << v_total_left(0) << ", " << v_total_left(1) << ", "
            << v_total_left(2) << ", " << v_total_right(0) << ", "
            << v_total_right(1) << ", " << v_total_right(2) << std::endl;
        } else {
          std::ofstream F("output.txt", std::ios::app);
          F << f_total_left(0) << ", " << f_total_left(1) << ", "
            << f_total_left(2) << "," << f_total_right(0) << ", "
            << f_total_right(1) << ", " << f_total_right(2) << ","
            << v_total_left(0) << ", " << v_total_left(1) << ", "
            << v_total_left(2) << ", " << v_total_right(0) << ", "
            << v_total_right(1) << ", " << v_total_right(2) << std::endl;
        }
      }
      // contact forces have been recorded to mpm and rigid
      // now advance mpm(small_dt)
      mpm::GridData<T> grid_data_scratch;
      mpm_solver_->AdvanceExplicitly(&result->mpm_state, *mpm_transfer_,
                                     deformable_model_->mpm_model(), mpm_dt,
                                     &grid_data_scratch, &mpm_scratch);
      mpm_transfer_->SetUpTransfer(&(result->mpm_state.sparse_grid),
                                   &(result->mpm_state.particles));
    }
  }

  void CalcGridDataFreeMotion(const systems::Context<T>& context,
                              mpm::GridData<T>* grid_data_free_motion) const {
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    double dt = manager_->plant().time_step();

    mpm::MpmSolverScratch<T>& mpm_scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.mpm_solver_scratch)
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();

    mpm_solver_->SolveGridVelocities(
        deformable_model_->mpm_model().newton_params(), state, *mpm_transfer_,
        deformable_model_->mpm_model(), dt, grid_data_free_motion,
        &mpm_scratch);
  }

  const mpm::GridData<T>& EvalGridDataFreeMotion(
      const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.grid_data_free_motion)
        .template Eval<mpm::GridData<T>>(context);
  }

  // TODO(zeshunzong): optimize it
  void CalcParticipatingVelocitiesMpm(const systems::Context<T>& context,
                                      VectorX<T>* result) const {
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    mpm::MpmSolverScratch<T>& mpm_scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.mpm_solver_scratch)
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();
    mpm::GridData<T> grid_data_prev_step;
    mpm_transfer_->P2G(state.particles, state.sparse_grid, &grid_data_prev_step,
                       &(mpm_scratch.transfer_scratch));
    if (!deformable_model_->MpmUseSchur()) {
      grid_data_prev_step.GetFlattenedVelocities(result);
    } else {
      const mpm::MpmGridNodesPermutation<T>& perm =
          EvalMpmGridNodesPermutation(context);
      grid_data_prev_step.ExtractVelocitiesFromIndices(perm.nodes_in_contact,
                                                       result);
    }
  }

  void CalcParticipatingFreeMotionVelocitiesMpm(
      const systems::Context<T>& context, VectorX<T>* result) const {
    const mpm::GridData<T>& grid_data_free_motion =
        EvalGridDataFreeMotion(context);
    if (!deformable_model_->MpmUseSchur()) {
      grid_data_free_motion.GetFlattenedVelocities(result);
    } else {
      const mpm::MpmGridNodesPermutation<T>& perm =
          EvalMpmGridNodesPermutation(context);
      grid_data_free_motion.ExtractVelocitiesFromIndices(perm.nodes_in_contact,
                                                         result);
    }
  }

  const mpm::GridData<T>& EvalGridDataPostContact(
      const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.grid_data_post_contact)
        .template Eval<mpm::GridData<T>>(context);
  }

  // TODO(zeshunzong): optimize it
  void CalcGridDataPostContact(const systems::Context<T>& context,
                               mpm::GridData<T>* grid_data_post_contact) const {
    // SAP takes the grid_data and handles contact
    const mpm::GridData<T>& grid_data_free_motion =
        EvalGridDataFreeMotion(context);
    (*grid_data_post_contact) = grid_data_free_motion;

    if (EvalMpmContactPairs(context).size() == 0) {
      // if no contact, no further treatment
      return;
    }
    contact_solvers::internal::ContactSolverResults<T> results =
        manager_->EvalContactSolverResults(context);
    if (!deformable_model_->MpmUseSchur()) {
      int mpm_dofs = grid_data_free_motion.num_active_nodes() * 3;
      VectorX<T> grid_v_post_contact_vec = results.v_next.tail(mpm_dofs);

      grid_data_post_contact->SetVelocities(grid_v_post_contact_vec);
    } else {
      // the velocities of the contact nodes are directly obtained from contact
      // results
      const mpm::MpmGridNodesPermutation<T>& perm =
          EvalMpmGridNodesPermutation(context);
      VectorX<T> in_contact_nodes_v_next =
          results.v_next.tail(perm.nodes_in_contact.size() * 3);
      // to compute the non_participating velocities,
      // non_participating_v_next = non_participating_v_free_motion +
      // schur.SolveForX(participating_v_next - participating_v_free_motion)

      const drake::multibody::contact_solvers::internal::SchurComplement&
          mpm_schur = EvalMpmTangentMatrixSchurComplement(context);

      VectorX<T> in_contact_nodes_v_free_motion;
      grid_data_free_motion.ExtractVelocitiesFromIndices(
          perm.nodes_in_contact, &in_contact_nodes_v_free_motion);
      VectorX<T> not_in_contact_nodes_v_free_motion;
      grid_data_free_motion.ExtractVelocitiesFromIndices(
          perm.nodes_not_in_contact, &not_in_contact_nodes_v_free_motion);

      VectorX<T> not_in_contact_nodes_v_next =
          not_in_contact_nodes_v_free_motion +
          mpm_schur.SolveForX(in_contact_nodes_v_next -
                              in_contact_nodes_v_free_motion);

      int count_in_contact = 0;
      int count_not_in_contact = 0;
      for (size_t i = 0; i < grid_data_free_motion.num_active_nodes(); ++i) {
        if (perm.nodes_in_contact.count(i)) {
          grid_data_post_contact->SetVelocityAt(
              in_contact_nodes_v_next.segment(3 * count_in_contact, 3), i);
          ++count_in_contact;
        } else {
          grid_data_post_contact->SetVelocityAt(
              not_in_contact_nodes_v_next.segment(3 * count_not_in_contact, 3),
              i);
          ++count_not_in_contact;
        }
      }
    }
  }

  void UpdateParticlesFromGridData(const systems::Context<T>& context,
                                   const mpm::GridData<T>& grid_data,
                                   mpm::Particles<T>* particles) const {
    double dt = manager_->plant().time_step();
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    mpm::MpmSolverScratch<T>& mpm_scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.mpm_solver_scratch)
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();
    mpm_transfer_->G2P(state.sparse_grid, grid_data, *particles,
                       &(mpm_scratch.particles_data),
                       &(mpm_scratch.transfer_scratch));
    // update velocity, F_trial, F_elastic, stress, B_matrix
    mpm_transfer_->UpdateParticlesState(mpm_scratch.particles_data, dt,
                                        particles);
    // update particle position, this is the last step
    particles->AdvectParticles(dt);
  }

  void WriteMpmParticlesToBgeo(const systems::Context<T>& context) const {
    double dt = manager_->plant().time_step();
    int current_step = std::round(context.get_time() / dt);
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    std::string output_filename =
        "./f" + std::to_string(current_step) + ".bgeo";
    mpm::internal::WriteParticlesToBgeo(
        output_filename, state.particles.positions(),
        state.particles.velocities(), state.particles.masses());

    // WriteAvgX(context, current_step);
    std::cout << "write " << output_filename << std::endl;
  }

  void WriteAvgX(const systems::Context<T>& context, int step) const {
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    double x_avg = 0.0;
    for (size_t p = 0; p < state.particles.num_particles(); ++p) {
      x_avg += state.particles.GetPositionAt(p)(0);
    }
    x_avg = x_avg / state.particles.num_particles();

    // Open a text file for writing
    if (step == 0) {
      std::ofstream outputFile("output.txt");
      outputFile << "mu is " << deformable_model_->mpm_model().friction_mu()
                 << std::endl;
      outputFile << x_avg << "," << std::endl;
      outputFile.close();
    } else {
      std::ofstream outputFile("output.txt", std::ios::app);
      outputFile << x_avg << "," << std::endl;
      outputFile.close();
    }
  }

  const std::vector<geometry::internal::MpmParticleContactPair<T>>&
  EvalMpmContactPairs(const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.mpm_contact_pairs)
        .template Eval<
            std::vector<geometry::internal::MpmParticleContactPair<T>>>(
            context);
  }

  void CalcMpmContactPairs(
      const systems::Context<T>& context,
      std::vector<geometry::internal::MpmParticleContactPair<T>>* result)
      const {
    DRAKE_ASSERT(result != nullptr);
    result->clear();
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    const geometry::QueryObject<T>& query_object =
        manager_->plant()
            .get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<T>>(context);
    // loop over each particle
    for (size_t p = 0; p < state.particles.num_particles(); ++p) {
      // compute the distance of this particle with each geometry in file
      std::vector<geometry::SignedDistanceToPoint<T>> p_to_geometries =
          query_object.ComputeSignedDistanceToPoint(
              state.particles.GetPositionAt(p));
      // identify those that are in contact, i.e. signed_distance < 0
      for (const auto& p2geometry : p_to_geometries) {
        if (p2geometry.distance < 0) {
          // if particle is inside rigid body, i.e. in contact
          // note: normal direction
          result->emplace_back(geometry::internal::MpmParticleContactPair<T>(
              p, p2geometry.id_G, p2geometry.distance,
              -p2geometry.grad_W.normalized(),
              state.particles.GetPositionAt(p)));
        }
      }
    }
  }

  void CalcMpmContactPairsWithParticlesInput(
      const systems::Context<T>& context,
      const mpm::Particles<T>& current_particles,
      std::vector<geometry::internal::MpmParticleContactPair<T>>* result)
      const {
    DRAKE_ASSERT(result != nullptr);
    result->clear();

    const geometry::QueryObject<T>& query_object =
        manager_->plant()
            .get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<T>>(context);
    // loop over each particle
    for (size_t p = 0; p < current_particles.num_particles(); ++p) {
      // compute the distance of this particle with each geometry in file
      std::vector<geometry::SignedDistanceToPoint<T>> p_to_geometries =
          query_object.ComputeSignedDistanceToPoint(
              current_particles.GetPositionAt(p));
      // identify those that are in contact, i.e. signed_distance < 0
      for (const auto& p2geometry : p_to_geometries) {
        if (p2geometry.distance < 0) {
          // if particle is inside rigid body, i.e. in contact
          // note: normal direction
          result->emplace_back(geometry::internal::MpmParticleContactPair<T>(
              p, p2geometry.id_G, p2geometry.distance,
              -p2geometry.grad_W.normalized(),
              current_particles.GetPositionAt(p)));
        }
      }
    }
  }

  const mpm::MpmGridNodesPermutation<T>& EvalMpmGridNodesPermutation(
      const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.mpm_grid_nodes_permutation)
        .template Eval<mpm::MpmGridNodesPermutation<T>>(context);
  }

  void CalcMpmGridNodesPermutation(
      const systems::Context<T>& context,
      mpm::MpmGridNodesPermutation<T>* result) const {
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    const std::vector<geometry::internal::MpmParticleContactPair<T>>&
        mpm_contact_pairs = EvalMpmContactPairs(context);
    result->Compute(mpm_contact_pairs, state);
  }

  // note: the linear dynamics matrix that sap needs is d2Edv2 in MPM
  void AppendLinearDynamicsMatrixMpm(const systems::Context<T>& context,
                                     std::vector<MatrixX<T>>* A) const {
    DRAKE_DEMAND(A != nullptr);

    if (!deformable_model_->MpmUseSchur()) {
      const mpm::MpmState<T>& state =
          context.template get_abstract_state<mpm::MpmState<T>>(
              deformable_model_->mpm_model().mpm_state_index());
      // TODO(zeshunzong): figure out how to remove the copy
      mpm::GridData<T> grid_data_free_motion_copy =
          EvalGridDataFreeMotion(context);
      mpm::DeformationState<T> deformation_state(
          state.particles, state.sparse_grid, grid_data_free_motion_copy);

      mpm::MpmSolverScratch<T>& mpm_scratch =
          manager_->plant()
              .get_cache_entry(cache_indexes_.mpm_solver_scratch)
              .get_mutable_cache_entry_value(context)
              .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();
      deformation_state.Update(*mpm_transfer_, manager_->plant().time_step(),
                               &(mpm_scratch),
                               (!deformable_model_->mpm_model()
                                     .newton_params()
                                     .linear_constitutive_model));
      MatrixX<T> linear_dynamics_matrix;
      deformable_model_->mpm_model().ComputeD2EnergyDV2(
          *mpm_transfer_, deformation_state, manager_->plant().time_step(),
          &linear_dynamics_matrix);

      A->emplace_back(std::move(linear_dynamics_matrix));
    } else {
      if (EvalMpmContactPairs(context).size() == 0) {
        // no grid nodes in contact
        A->emplace_back(MatrixX<double>(0, 0));
      } else {
        A->emplace_back(
            EvalMpmTangentMatrixSchurComplement(context).get_D_complement());
      }
    }
  }

  const drake::multibody::contact_solvers::internal::SchurComplement&
  EvalMpmTangentMatrixSchurComplement(
      const systems::Context<T>& context) const {
    return manager_->plant()
        .get_cache_entry(cache_indexes_.mpm_tangent_matrix_schur_complement)
        .template Eval<
            drake::multibody::contact_solvers::internal::SchurComplement>(
            context);
  }

  void CalcMpmTangentMatrixSchurComplement(
      const systems::Context<T>& context,
      drake::multibody::contact_solvers::internal::SchurComplement* result)
      const {
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    // TODO(zeshunzong): figure out how to remove the copy
    mpm::GridData<T> grid_data_free_motion_copy =
        EvalGridDataFreeMotion(context);
    mpm::DeformationState<T> deformation_state(
        state.particles, state.sparse_grid, grid_data_free_motion_copy);

    mpm::MpmSolverScratch<T>& mpm_scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.mpm_solver_scratch)
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<mpm::MpmSolverScratch<T>>();
    deformation_state.Update(*mpm_transfer_, manager_->plant().time_step(),
                             &(mpm_scratch),
                             (!deformable_model_->mpm_model()
                                   .newton_params()
                                   .linear_constitutive_model));
    multibody::contact_solvers::internal::
        BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>
            hessian = deformable_model_->mpm_model()
                          .ComputeD2EnergyDV2SymmetricBlockSparse(
                              *mpm_transfer_, deformation_state,
                              manager_->plant().time_step());
    const mpm::MpmGridNodesPermutation<T>& mpm_grid_nodes_permutation =
        EvalMpmGridNodesPermutation(context);
    (*result) = drake::multibody::contact_solvers::internal::SchurComplement(
        hessian, mpm_grid_nodes_permutation.nodes_not_in_contact);
  }

  void AppendDiscreteContactPairsMpm(
      const systems::Context<T>& context,
      DiscreteContactData<DiscreteContactPair<T>>* result) const {
    const std::vector<geometry::internal::MpmParticleContactPair<T>>&
        mpm_contact_pairs = EvalMpmContactPairs(context);
    geometry::GeometryId dummy_id = geometry::GeometryId::get_new_id();
    // TODO(zeshunzong): don't know how to use
    std::cout << "num of mpm contact pairs: " << mpm_contact_pairs.size()
              << std::endl;
    for (const auto& mpm_contact_pair : mpm_contact_pairs) {
      const T d = deformable_model_->mpm_damping();  // damping
      const T k = deformable_model_->mpm_stiffness();
      const T fn0 = k * std::abs(mpm_contact_pair.penetration_distance);
      const T tau = NAN;
      result->AppendDeformableData(DiscreteContactPair<T>{
          dummy_id, mpm_contact_pair.non_mpm_id,
          mpm_contact_pair.particle_in_contact_position,
          mpm_contact_pair.normal, mpm_contact_pair.penetration_distance, fn0,
          k, d, tau, deformable_model_->mpm_model().friction_mu()});
    }
  }

  void AppendContactKinematicsMpm(
      const systems::Context<T>& context,
      DiscreteContactData<ContactPairKinematics<T>>* result) const {
    DRAKE_DEMAND(result != nullptr);
    const mpm::MpmState<T>& state =
        context.template get_abstract_state<mpm::MpmState<T>>(
            deformable_model_->mpm_model().mpm_state_index());
    const MultibodyTreeTopology& tree_topology =
        manager_->internal_tree().get_topology();
    DRAKE_DEMAND(num_deformable_bodies() == 0);  // note: no FEM right now!
    const TreeIndex clique_index_mpm(tree_topology.num_trees() +
                                     num_deformable_bodies());

    const std::vector<geometry::internal::MpmParticleContactPair<T>>&
        mpm_contact_pairs = EvalMpmContactPairs(context);
    const int nv = manager_->plant().num_velocities();

    if (mpm_contact_pairs.size() == 0) {
      return;
    }

    for (auto& contact_pair : mpm_contact_pairs) {
      Vector3<T> vn = Vector3<T>::Zero();

      // for each contact pair, want J = R_CW * Jacobian_block = R_CW *
      // [-Jmpm | Jrigid]

      /* Compute the rotation matrix R_CW */
      constexpr int kZAxis = 2;
      math::RotationMatrix<T> R_WC =
          math::RotationMatrix<T>::MakeFromOneUnitVector(contact_pair.normal,
                                                         kZAxis);
      const math::RotationMatrix<T> R_CW = R_WC.transpose();

      /* We have at most two blocks per contact. */
      std::vector<typename ContactPairKinematics<T>::JacobianTreeBlock>
          jacobian_blocks;
      jacobian_blocks.reserve(2);

      /* MPM part of Jacobian, note this is -J_mpm */
      if (!deformable_model_->MpmUseSchur()) {
        MatrixX<T> J_mpm;
        mpm_transfer_->CalcJacobianGridVToParticleVAtParticle(
            contact_pair.particle_in_contact_index, state.particles,
            state.sparse_grid, &J_mpm);

        J_mpm = -R_CW.matrix() * J_mpm;

        jacobian_blocks.emplace_back(
            clique_index_mpm,
            contact_solvers::internal::MatrixBlock<T>(std::move(J_mpm)));
      } else {
        contact_solvers::internal::Block3x3SparseMatrix<T> J_mpm =
            mpm_transfer_
                ->CalcRTimesJacobianGridVToParticleVWithPermutationAtParticle(
                    -R_CW.matrix(), contact_pair.particle_in_contact_index,
                    state, EvalMpmGridNodesPermutation(context));
        jacobian_blocks.emplace_back(
            clique_index_mpm,
            contact_solvers::internal::MatrixBlock<T>(std::move(J_mpm)));
      }

      vn += R_CW.matrix() * state.particles.GetVelocityAt(
                                contact_pair.particle_in_contact_index);

      /* Non-MPM (rigid) part of Jacobian */
      const BodyIndex index_B =
          manager_->geometry_id_to_body_index().at(contact_pair.non_mpm_id);
      const TreeIndex tree_index_rigid =
          tree_topology.body_to_tree_index(index_B);

      if (tree_index_rigid.is_valid()) {
        Matrix3X<T> Jv_v_WBc_W(3, nv);
        const Body<T>& rigid_body = manager_->plant().get_body(index_B);
        const Frame<T>& frame_W = manager_->plant().world_frame();
        manager_->internal_tree().CalcJacobianTranslationalVelocity(
            context, JacobianWrtVariable::kV, rigid_body.body_frame(), frame_W,
            contact_pair.particle_in_contact_position, frame_W, frame_W,
            &Jv_v_WBc_W);
        Matrix3X<T> J_rigid =
            R_CW.matrix() *
            Jv_v_WBc_W.middleCols(
                tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                tree_topology.num_tree_velocities(tree_index_rigid));
        jacobian_blocks.emplace_back(
            tree_index_rigid,
            contact_solvers::internal::MatrixBlock<T>(std::move(J_rigid)));

        const Eigen::VectorBlock<const VectorX<T>> v =
            manager_->plant().GetVelocities(context);
        vn -= J_rigid *
              v.segment(
                  tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                  tree_topology.num_tree_velocities(tree_index_rigid));
      }

      // configuration part
      const int objectA =
          tree_topology.num_trees() + num_deformable_bodies() + 1;
      const int objectB = index_B;  // rigid body

      // Contact point position relative to rigid body B, same as in FEM-rigid
      const math::RigidTransform<T>& X_WB =
          manager_->plant().EvalBodyPoseInWorld(
              context, manager_->plant().get_body(index_B));
      const Vector3<T>& p_WB = X_WB.translation();
      const Vector3<T> p_BC_W =
          contact_pair.particle_in_contact_position - p_WB;

      const Vector3<T> p_ApC_W{NAN, NAN, NAN};

      // TODO(zeshunzong): temporarily unused
      contact_solvers::internal::ContactConfiguration<T> configuration{
          .objectA = objectA,
          .p_ApC_W = p_ApC_W,  // to be changed, see above
          .objectB = objectB,
          .p_BqC_W = p_BC_W,
          .phi = contact_pair.penetration_distance,
          .vn = -vn(2),
          .fe = std::abs(contact_pair.penetration_distance) *
                deformable_model_->mpm_stiffness(),
          .R_WC = R_WC};
      // TODO(zeshunzong): we are not distinguishing between fem and mpm rn
      result->AppendDeformableData(ContactPairKinematics<T>(
          std::move(jacobian_blocks), std::move(configuration)));
    }
  }

  // ------------ newly added for MPM ---------------

  ~DeformableDriver();

  int num_deformable_bodies() const { return deformable_model_->num_bodies(); }

  // TODO(xuchenhan-tri): Implement CloneToDouble() and allow cloning to
  // double.
  bool is_cloneable_to_double() const final { return false; }
  bool is_cloneable_to_autodiff() const final { return false; }
  bool is_cloneable_to_symbolic() const final { return false; }

  /* Declares cache entries used by this DeformableDriver through the given
   manager.
   @pre `manager` is not nullptr and points to the same DiscreteUpdateManager
   provided at construction. */
  void DeclareCacheEntries(DiscreteUpdateManager<T>* manager);

  /* Evaluates the velocities of all participating dofs. See class
   documentation for how the velocities are ordered. */
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
      DiscreteContactData<DiscreteContactPair<T>>* pairs) const;

  /* Appends the contact kinematics information for each contact pair where at
   least one of the body in contact is deformable.
   @pre result != nullptr. */
  void AppendContactKinematics(
      const systems::Context<T>& context,
      DiscreteContactData<ContactPairKinematics<T>>* result) const;

  /* Appends the constraint kinematics information for each deformable rigid
   fixed constraint.
   @pre result != nullptr. */
  void AppendDeformableRigidFixedConstraintKinematics(
      const systems::Context<T>& context,
      std::vector<contact_solvers::internal::FixedConstraintKinematics<T>>*
          result) const;

  /* Computes the contact information for all deformable bodies for the given
   `context`.
   @pre contact_info != nullptr. */
  void CalcDeformableContactInfo(
      const systems::Context<T>& context,
      std::vector<DeformableContactInfo<T>>* contact_info) const;

  /* Evaluates FemState at the next time step for each deformable body and
   copies the them into the corresponding DiscreteValues.
   @pre next_states != nullptr. */
  void CalcDiscreteStates(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const;

  /* Evaluates the multiplexer for participating velocities for all bodies.
   @pre result != nullptr. */
  const Multiplexer<T>& EvalParticipatingVelocityMultiplexer(
      const systems::Context<T>& context) const;

  /* Evaluates the constraint participation information of the deformable body
   with the given `index`. See geometry::internal::ContactParticipation. */
  const geometry::internal::ContactParticipation& EvalConstraintParticipation(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  const DeformableModel<T>* deformable_model() const {
    return deformable_model_;
  };

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
    std::vector<systems::CacheIndex> fem_solvers;
    std::vector<systems::CacheIndex> next_fem_states;
    systems::CacheIndex deformable_contact;
    std::vector<systems::CacheIndex> constraint_participations;
    std::vector<systems::CacheIndex> dof_permutations;
    std::unordered_map<geometry::GeometryId, systems::CacheIndex>
        vertex_permutations;
    systems::CacheIndex participating_velocity_mux;
    systems::CacheIndex participating_velocities;
    systems::CacheIndex participating_free_motion_velocities;

    // mpm cache indexes
    systems::CacheIndex mpm_post_contact_result;
    systems::CacheIndex mpm_solver_scratch;
    systems::CacheIndex grid_data_free_motion;
    systems::CacheIndex grid_data_post_contact;
    systems::CacheIndex mpm_contact_pairs;
    // systems::CacheIndex grid_indices_in_contact;
    // systems::CacheIndex grid_indices_not_in_contact;
    systems::CacheIndex mpm_grid_nodes_permutation;
    systems::CacheIndex mpm_tangent_matrix_schur_complement;
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

  /* Given the state of the deformable body with `index` in the given
   `context`, computes its "free motion" state (the state the body would have
   at the next time step in the absence of contact or constraints) and the
   dependent Schur complement of the tangent matrix of the FEM model.
   @pre state_and_data != nullptr and is compatible with the FemModel
   associated with the deformable body with the given `index`. */
  void CalcFreeMotionFemSolver(const systems::Context<T>& context,
                               DeformableBodyIndex index,
                               fem::internal::FemSolver<T>* fem_solver) const;

  /* Eval version of CalcFreeMotionFemState(). */
  const fem::internal::FemSolver<T>& EvalFreeMotionFemSolver(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  const fem::FemState<T>& EvalFreeMotionFemState(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  const contact_solvers::internal::SchurComplement&
  EvalFreeMotionTangentMatrixSchurComplement(const systems::Context<T>& context,
                                             DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `index` in the given
   `context`, computes the state of the deformable body at the next time step.
   @note The state of the deformable body will the same as the "free motion"
         state in the absence of contact or constraints. Otherwise, the
   discrete solver results for participating dofs are evaluated, and the Schur
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

  /* Computes the contact information for all registered deformable bodies
   @pre The geometry query input port of the MultibodyPlant that owns the
        manager associated with this DeformableDriver is connected.
   @pre result != nullptr. */
  void CalcDeformableContact(
      const systems::Context<T>& context,
      geometry::internal::DeformableContact<T>* result) const;

  /* Eval version of CalcDeformableContact(). */
  const geometry::internal::DeformableContact<T>& EvalDeformableContact(
      const systems::Context<T>& context) const;

  /* Calc version of EvalConstraintParticipation.
   @pre constraint_participation != nullptr. */
  void CalcConstraintParticipation(
      const systems::Context<T>& context, DeformableBodyIndex index,
      geometry::internal::ContactParticipation* constraint_participation) const;

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

  CacheIndexes cache_indexes_;
  /* Modeling information about all deformable bodies. */
  const DeformableModel<T>* const deformable_model_;
  const DiscreteUpdateManager<T>* const manager_;
  /* The integrator used to advance deformable body free motion states in
   time. */
  std::unique_ptr<fem::internal::DiscreteTimeIntegrator<T>> integrator_;

  // mpm stuff
  std::unique_ptr<mpm::MpmTransfer<T>> mpm_transfer_;
  std::unique_ptr<mpm::MpmSolver<T>> mpm_solver_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
