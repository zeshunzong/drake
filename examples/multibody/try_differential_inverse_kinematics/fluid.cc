#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/collision_filter_manager.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat_point_cloud_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/scene_graph_inspector.h"
#include "drake/manipulation/kuka_iiwa/iiwa_constants.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/inverse_kinematics/differential_inverse_kinematics_integrator.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_with_plasticity.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/multiplexer.h"

DEFINE_double(simulation_time, 2.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.1, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(friction, 0.1, "mpm friction");
DEFINE_double(ppc, 1, "mpm ppc");
DEFINE_double(shift, 0.98, "shift");
DEFINE_double(damping, 10.0, "larger, more damping");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::manipulation::kuka_iiwa::get_iiwa_max_joint_velocities;
using drake::math::RigidTransformd;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::DifferentialInverseKinematicsIntegrator;
using drake::multibody::DifferentialInverseKinematicsParameters;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

RigidTransformd FromXyzRpy(const Vector3<double>& rpy,
                           const Vector3<double>& p) {
  return RigidTransformd(math::RollPitchYaw<double>(rpy), p);
}
RigidTransformd FromXyzRpyDegree(const Vector3<double>& rpy_deg,
                                 const Vector3<double>& p) {
  return RigidTransformd(
      math::RollPitchYaw<double>(rpy_deg * 3.1415926 / 180.0), p);
}

class HandPoseController : public drake::systems::LeafSystem<double> {
 public:
  HandPoseController(const multibody::MultibodyPlant<double>& plant)
      : plant_(plant) {
    this->DeclareVectorOutputPort(
        "AllegroDesiredState", drake::systems::BasicVector<double>(size_),
        &HandPoseController::CalcDesiredState, {this->time_ticket()});
  }

  void CalcDesiredState(const Context<double>& context,
                        drake::systems::BasicVector<double>* output) const {
    double alpha = 1.0;

    if (context.get_time() < prep_time_) {
      Eigen::VectorXd positions = GetHomePosition();
      Eigen::VectorXd q_and_v(32);
      q_and_v << positions, GetHomeVelocity();
      output->set_value(q_and_v);
    } else if ((context.get_time() > grip_start_) &&
               (context.get_time() < grip_start_ + grip_time_)) {
      // start gripping
      double t = (context.get_time() - grip_start_) / grip_time_;
      Eigen::VectorXd positions =
          std::max(1.0 - t * alpha, 0.0) * GetHomePosition() +
          std::min(t * alpha, 1.0) * GetGripPosition();
      Eigen::VectorXd q_and_v(32);
      q_and_v << positions, GetHomeVelocity();
      output->set_value(q_and_v);
    } else if ((context.get_time() > grip_release_) &&
               (context.get_time() < grip_release_ + grip_time_)) {
      double t = (context.get_time() - grip_release_) / grip_time_;
      Eigen::VectorXd positions =
          std::max(1.0 - t * alpha, 0.0) * GetGripPosition() +
          std::min(t * alpha, 1.0) * GetHomePosition();
      Eigen::VectorXd q_and_v(32);
      q_and_v << positions, GetHomeVelocity();
      output->set_value(q_and_v);
    }
  }

  Eigen::VectorXd GetHomePosition() const {
    Eigen::VectorXd pos(16);
    pos.setZero();
    pos(0) = 1.4;
    pos(1) = 0.25;
    return pos;
  }

  Eigen::VectorXd GetHomeVelocity() const { return Eigen::VectorXd::Zero(16); }

  Eigen::VectorXd GetGripPosition() const {
    Eigen::VectorXd vec(16);
    vec << 1.4, 0.25, 0.26, 1.22, -0.11, 0.54, 0.88, 0.93, 0.0, 0.54, 0.88,
        0.93, 0.12, 0.54, 0.88, 0.93;
    return (vec);
  }

  std::vector<std::string> GetPreferredJointOrdering() {
    std::vector<std::string> joint_name_mapping;
    // Thumb finger
    joint_name_mapping.push_back("joint_12");
    joint_name_mapping.push_back("joint_13");
    joint_name_mapping.push_back("joint_14");
    joint_name_mapping.push_back("joint_15");
    // Index finger
    joint_name_mapping.push_back("joint_0");
    joint_name_mapping.push_back("joint_1");
    joint_name_mapping.push_back("joint_2");
    joint_name_mapping.push_back("joint_3");
    // Middle finger
    joint_name_mapping.push_back("joint_4");
    joint_name_mapping.push_back("joint_5");
    joint_name_mapping.push_back("joint_6");
    joint_name_mapping.push_back("joint_7");
    // Ring finger
    joint_name_mapping.push_back("joint_8");
    joint_name_mapping.push_back("joint_9");
    joint_name_mapping.push_back("joint_10");
    joint_name_mapping.push_back("joint_11");
    return joint_name_mapping;
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  int size_ = 16 * 2;  // 4 fingers with 4 joints
  double grip_time_ = 0.3;
  double prep_time_ = 0.4;
  double grip_start_ = 0.5;
  double grip_release_ = 4.99;
};

class IiwaController : public drake::systems::LeafSystem<double> {
 public:
  IiwaController(const multibody::MultibodyPlant<double>& plant,
                 const RigidTransformd& init_pose)
      : plant_(plant) {
    robot_state_index_ =
        this->DeclareVectorInputPort("robot_state", 14).get_index();

    this->DeclareAbstractOutputPort("X_WG_desired", init_pose,
                                    &IiwaController::CalcOutput);
    VectorXd X_xyz_rpy = Eigen::VectorXd::Zero(6);
    // state = [ryp, translation]
    X_xyz_rpy.segment(0, 3) = init_pose.rotation().ToRollPitchYaw().vector();
    X_xyz_rpy.segment(3, 3) = init_pose.translation();
    this->DeclareDiscreteState(X_xyz_rpy);
    this->DeclarePeriodicDiscreteUpdateEvent(plant_.time_step(), 0,
                                             &IiwaController::Update);
  }

  const systems::InputPort<double>& robot_state_input_port() const {
    return this->get_input_port(robot_state_index_);
  }

  void CalcOutput(const Context<double>& context,
                  RigidTransformd* value) const {
    // Eigen::VectorXd current_robot_qv =
    //     this->get_input_port(robot_state_index_).Eval(context);
    const VectorXd xd = context.get_discrete_state().value();
    // xd = [rpy, translation]
    *value = FromXyzRpy(xd.segment(0, 3), xd.segment(3, 3));
  }

  void Update(const Context<double>& context,
              systems::DiscreteValues<double>* next_states) const {
    const VectorX<double>& current_state_values =
        context.get_discrete_state().value();
    unused(current_state_values);

    // fake update:
    VectorX<double> dX = current_state_values;
    dX.setZero();
    if ((context.get_time() >= 0.0) && (context.get_time() < 0.1)) {
      dX(4) = 0.002;  // vertical
    }
    if ((context.get_time() > 0.8) && (context.get_time() < 1.2)) {
      dX(5) = 0.006;  // vertical
    }
    if ((context.get_time() > 1.2) && (context.get_time() < 1.6)) {
      dX(1) = 0.065;  // rotate
    }

    // // dX(0) = -0.004; // r
    // // dX(1) = -0.004; // p
    // // dX(2) = -0.004;  // y
    // // dX(3) = -0.004; // x
    // // dX(4) = 0.004;  // y
    // dX(5) = 0.00;  // vertical
    auto new_value = current_state_values + dX;
    next_states->set_value(new_value);
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  int robot_state_index_{};

  Eigen::Vector3d trans_;
  Eigen::Vector3d rpy_;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = "lagged";
  // plant_config.contact_model = "hydroelastic";
  plant_config.stiction_tolerance = 1e-5;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties compliant_hydro_props;
  ProximityProperties rigid_proximity_props;

  const CoulombFriction<double> surface_friction(100.0, 100.0);
  AddContactMaterial(FLAGS_damping, {}, surface_friction,
                     &compliant_hydro_props);
  AddContactMaterial(FLAGS_damping, {}, surface_friction,
                     &rigid_proximity_props);

  AddCompliantHydroelasticProperties(0.01, 1e6, &compliant_hydro_props);
  AddRigidHydroelasticProperties(0.01, &rigid_proximity_props);

  Box ground{20, 20, 10};
  const RigidTransformd X_WGround(Eigen::Vector3d{0, 0, -5});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WGround, ground,
                                  "ground_collision", compliant_hydro_props);
  IllustrationProperties illustration_props_G;
  illustration_props_G.AddProperty("phong", "diffuse",
                                   Vector4d(0.9, 0.9, 0.9, 1.0));
  plant.RegisterVisualGeometry(plant.world_body(), X_WGround, ground,
                               "ground_visual",
                               std::move(illustration_props_G));

  double mug_outer_radius = 0.055;
  double mug_height = 3.5 * mug_outer_radius;
  double mug_thickness = mug_outer_radius / 6.0;
  double mug_inner_radius = mug_outer_radius - mug_thickness;
  double mug_inner_height = mug_height - mug_thickness;
  Vector4<double> outer_cylinder_color(1.0, 0.55, 0.0, 0.2);
  Vector4<double> inner_cylinder_color(1.0, 0.55, 0.5, 0.6);

  Vector3<double> mug_outer_translation(0.69 + 0.0, 0.08, mug_height / 2.0);

  const SpatialInertia<double> outer_cylinder_spatial =
      SpatialInertia<double>::SolidCylinderWithDensity(
          10, mug_height, mug_outer_radius, Vector3<double>(0, 0, 1));
  const RigidBody<double>& outer_cylinder_body =
      plant.AddRigidBody("outer_cylinder", outer_cylinder_spatial);
  plant.RegisterVisualGeometry(
      outer_cylinder_body, RigidTransformd::Identity(),
      drake::geometry::Cylinder(mug_outer_radius, mug_height),
      "OuterCylinderVisual", outer_cylinder_color);
  geometry::GeometryId outer_cylinder_geometry_id =
      plant.RegisterCollisionGeometry(
          outer_cylinder_body, RigidTransformd::Identity(),
          drake::geometry::Cylinder(mug_outer_radius, mug_height),
          "OuterCylinderCollision", compliant_hydro_props);
  geometry::GeometryId inner_cylinder_geometry_id =
      plant.RegisterCollisionGeometry(
          outer_cylinder_body,
          RigidTransformd(Eigen::Vector3d{0, 0, mug_thickness / 2.0}),
          drake::geometry::Cylinder(mug_inner_radius, mug_inner_height),
          "InnerCylinderCollision", compliant_hydro_props);
  plant.RegisterVisualGeometry(
      plant.GetBodyByName("outer_cylinder"),
      RigidTransformd(Eigen::Vector3d{0, 0, mug_thickness / 2.0}),
      drake::geometry::Cylinder(mug_inner_radius, mug_inner_height),
      "InnerCylinderVisual", inner_cylinder_color);

//   // ------------- a container on the ground
//   IllustrationProperties illustration_props;
//   illustration_props.AddProperty("phong", "diffuse",
//                                  Vector4d(0.7, 0.5, 0.4, 0.3));
//   Vector3<double> shift(0, 0.5, 0);
//   double pad_size = 0.15;
//   Box vertical_pad1{mug_thickness, pad_size, pad_size};
//   plant.RegisterCollisionGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d((pad_size) / 2.0,
//                                       -mug_thickness / 2.0, pad_size / 2.0) + shift),
//       vertical_pad1, "vertical_pad1", rigid_proximity_props);
//   plant.RegisterVisualGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d((pad_size) / 2.0,
//                                       -mug_thickness / 2.0, pad_size / 2.0)+ shift),
//       vertical_pad1, "vertical_pad1_visual", (illustration_props));
//   Box vertical_pad2{pad_size, mug_thickness, pad_size};
//   plant.RegisterCollisionGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(mug_thickness / 2.0, (pad_size) / 2.0,
//                                       pad_size / 2.0)+ shift),
//       vertical_pad2, "vertical_pad2_pad", rigid_proximity_props);
//   plant.RegisterVisualGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(mug_thickness / 2.0, (pad_size) / 2.0,
//                                       pad_size / 2.0)+ shift),
//       vertical_pad2, "vertical_pad2_visual", (illustration_props));
//   Box vertical_pad3{mug_thickness, pad_size, pad_size};
//   plant.RegisterCollisionGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(-(pad_size) / 2.0,
//                                       mug_thickness / 2.0, pad_size / 2.0)+ shift),
//       vertical_pad3, "vertical_pad3_pad", rigid_proximity_props);
//   plant.RegisterVisualGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(-(pad_size) / 2.0,
//                                       mug_thickness / 2.0, pad_size / 2.0)+ shift),
//       vertical_pad3, "vertical_pad3_visual", (illustration_props));

//   Box vertical_pad4{pad_size, mug_thickness, pad_size};
//   plant.RegisterCollisionGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(-mug_thickness / 2.0,
//                                       -(pad_size) / 2.0, pad_size / 2.0)+ shift),
//       vertical_pad4, "vertical_pad4", rigid_proximity_props);
//   plant.RegisterVisualGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(-mug_thickness / 2.0,
//                                       -(pad_size) / 2.0, pad_size / 2.0)+ shift),
//       vertical_pad4, "vertical_pad4_visual", (illustration_props));

//   Box bottom_pad{pad_size + mug_thickness, pad_size + mug_thickness,
//                  mug_thickness};
//   plant.RegisterCollisionGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(0, 0, mug_thickness / 2.0)+ shift), bottom_pad,
//       "bottom_pad", rigid_proximity_props);
//   plant.RegisterVisualGeometry(
//       plant.world_body(),
//       RigidTransformd(Eigen::Vector3d(0, 0, mug_thickness / 2.0)+ shift), bottom_pad,
//       "bottom_pad_visual", (illustration_props));
//   // ------------- a container on the ground

  MultibodyPlant<double> iiwa_controller_plant =
      MultibodyPlant<double>(plant_config.time_step);

  // plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());
  multibody::Parser parser(&plant);
  const std::string filename = FindResourceOrThrow(
      "drake/manipulation/models/"
      "iiwa_description/iiwa7/iiwa7_no_collision.sdf");
  auto iiwa = parser.AddModels(filename)[0];
  Parser(&iiwa_controller_plant).AddModels(filename);

  std::string hand_filename = FindResourceOrThrow(
      "drake/manipulation/models/"
      "allegro_hand_description/sdf/allegro_hand_description_right.sdf");
  auto allegro = parser.AddModels(hand_filename)[0];

  RigidTransformd iiwa_position(Eigen::Vector3d(0, 0, 0));

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("iiwa_link_0").body_frame(),
                   iiwa_position);
  iiwa_controller_plant.WeldFrames(
      iiwa_controller_plant.world_frame(),
      iiwa_controller_plant.GetBodyByName("iiwa_link_0").body_frame(),
      iiwa_position);

  plant.WeldFrames(
      plant.GetBodyByName("iiwa_link_7").body_frame(),
      plant.GetBodyByName("hand_root").body_frame(),
      FromXyzRpyDegree(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.0, 0, 0.0)));

  // mpm stuff
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set =
          std::make_unique<drake::multibody::mpm::internal::CylinderLevelSet>(
              mug_height / 2.0, mug_inner_radius * 0.98);
  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model = std::make_unique<drake::multibody::mpm::constitutive_model::
                                   EquationOfState<double>>(
          0.111, 0.111, 100.0, 7.0);
  Vector3<double> translation = {mug_outer_translation(0),
                                 mug_outer_translation(1),
                                 (mug_height-mug_thickness) / 2.0 + mug_thickness * 1.0};
  std::unique_ptr<math::RigidTransform<double>> pose =
      std::make_unique<math::RigidTransform<double>>(translation);
  double h = mug_outer_radius / 2.4;
  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set),
                                          std::move(model), std::move(pose),
                                          1000.0, h);

  owned_deformable_model->SetMpmMinParticlesPerCell(
      static_cast<int>(FLAGS_ppc));
  owned_deformable_model->SetMpmFriction(FLAGS_friction);
  owned_deformable_model->SetMpmDamping(FLAGS_damping);
  owned_deformable_model->SetMpmStiffness(1e5);

  owned_deformable_model->inner_cylinder_id_ = inner_cylinder_geometry_id;
  owned_deformable_model->outer_cylinder_id_ = outer_cylinder_geometry_id;

  plant.AddPhysicalModel(std::move(owned_deformable_model));

  double Kp = 1000000.0;
  double Kd = 2 * std::sqrt(Kp);

  drake::multibody::PdControllerGains gain(Kp, Kd);
  for (int i = 0; i < plant.num_actuators(); ++i) {
    plant.get_mutable_joint_actuator(drake::multibody::JointActuatorIndex(i))
        .set_controller_gains(gain);
  }

  plant.Finalize();
  iiwa_controller_plant.Finalize();

  Eigen::VectorXd iiwa_initial_joint_values(7);
  iiwa_initial_joint_values << 0, 1.2, 0, -1.6, 0, -1.2, 1.57;
  // Eigen::VectorXd iiwa_velocity_limits(7);
  // iiwa_velocity_limits << 1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3;

  // hand controller
  auto hand_pose_controller = builder.template AddSystem<HandPoseController>(
      plant);  // put gripper timing here
  std::vector<std::string> preferred_joint_ordering =
      hand_pose_controller->GetPreferredJointOrdering();

  int nu_allegro = plant.num_actuated_dofs(allegro);
  int nq_allegro = plant.num_positions(allegro);
  int nv_allegro = plant.num_velocities(allegro);
  unused(nv_allegro);

  Eigen::MatrixXd state_selector =
      Eigen::MatrixXd::Zero(2 * nu_allegro, 2 * nu_allegro);
  int u = 0;
  std::vector<std::string> actuated_joints;
  for (int a = 0; a < plant.num_actuators(); ++a) {
    auto& actuator =
        plant.get_joint_actuator(drake::multibody::JointActuatorIndex(a));
    for (int i = 0; i < static_cast<int>(preferred_joint_ordering.size());
         ++i) {
      if (actuator.joint().name() == preferred_joint_ordering[i]) {
        actuated_joints.push_back(actuator.joint().name());
        state_selector(u, i) = 1;
        state_selector(nu_allegro + u, nq_allegro + i) = 1;
        ++u;
        break;
      }
    }
  }
  auto actuated_states_selector =
      builder.template AddSystem<drake::systems::MatrixGain<double>>(
          state_selector);
  builder.Connect(hand_pose_controller->get_output_port(),
                  actuated_states_selector->get_input_port());
  builder.Connect(actuated_states_selector->get_output_port(),
                  plant.get_desired_state_input_port(allegro));

  // iiwa controller
  // first find the initial pose for the current joint values
  std::unique_ptr<Context<double>> temp_context = plant.CreateDefaultContext();
  plant.SetPositions(temp_context.get(), iiwa, iiwa_initial_joint_values);
  RigidTransformd iiwa_controller_initial_pose =
      plant.GetBodyByName("iiwa_link_7")
          .body_frame()
          .CalcPoseInWorld(*(temp_context.get()));

  auto iiwa_controller = builder.template AddSystem<IiwaController>(
      plant, iiwa_controller_initial_pose);
  int nq_iiwa = plant.num_positions(iiwa);
  int nv_iiwa = plant.num_velocities(iiwa);
  int nu_iiwa = plant.num_actuated_dofs(iiwa);
  unused(nu_iiwa);

  DifferentialInverseKinematicsParameters params(nq_iiwa, nv_iiwa);
  params.set_nominal_joint_position(iiwa_initial_joint_values);

  auto diff_ik =
      builder.template AddSystem<DifferentialInverseKinematicsIntegrator>(
          iiwa_controller_plant,
          iiwa_controller_plant.GetFrameByName("iiwa_link_7"),
          plant_config.time_step, params);

  std::vector<int> input_sizes;
  input_sizes.push_back(nq_iiwa);
  input_sizes.push_back(nv_iiwa);
  auto mux = builder.template AddSystem<drake::systems::Multiplexer<double>>(
      input_sizes);
  auto zero_vs =
      builder.template AddSystem<drake::systems::ConstantVectorSource>(
          Eigen::VectorXd::Zero(nv_iiwa));
  builder.Connect(plant.get_state_output_port(iiwa),
                  iiwa_controller->robot_state_input_port());
  builder.Connect(iiwa_controller->get_output_port(),
                  diff_ik->GetInputPort("X_WE_desired"));
  builder.Connect(plant.get_state_output_port(iiwa),
                  diff_ik->GetInputPort("robot_state"));
  builder.Connect(diff_ik->GetOutputPort("joint_positions"),
                  mux->get_input_port(0));
  builder.Connect(zero_vs->get_output_port(), mux->get_input_port(1));
  builder.Connect(mux->get_output_port(),
                  plant.get_desired_state_input_port(iiwa));

  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.publish_period = FLAGS_time_step;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, meshcat_params);
  auto meshcat_pc_visualizer =
      builder.AddSystem<drake::geometry::MeshcatPointCloudVisualizer>(
          meshcat, "cloud", meshcat_params.publish_period);
  meshcat_pc_visualizer->set_point_size(0.01);
  builder.Connect(deformable_model->mpm_point_cloud_port(),
                  meshcat_pc_visualizer->cloud_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);

  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("outer_cylinder"),
                        RigidTransformd(mug_outer_translation));
  //   auto& diff_ik_context =
  //       diff_ik->GetMyMutableContextFromRoot(&mutable_context);

  plant.SetPositions(&plant_context, iiwa, iiwa_initial_joint_values);

  Eigen::VectorXd allegro_pos(16);
  allegro_pos.setZero();
  allegro_pos(0) = 1.4;
  allegro_pos(1) = 0.25;

  plant.SetPositions(&plant_context, allegro, allegro_pos);

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  sleep(6);
  bool recording = true;

  if (recording) {
    meshcat->StartRecording();
    simulator.AdvanceTo(FLAGS_simulation_time);
    meshcat->StopRecording();
    meshcat->PublishRecording();

  } else {
    simulator.AdvanceTo(FLAGS_simulation_time);
  }

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A simple parallel gripper grasps a deformable torus on the ground, "
      "lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
