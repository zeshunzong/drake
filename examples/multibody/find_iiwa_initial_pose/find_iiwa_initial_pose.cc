#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/manipulation/kuka_iiwa/iiwa_constants.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/inverse_kinematics/differential_inverse_kinematics_integrator.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
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

DEFINE_double(simulation_time, 1.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.1, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(friction, 0.0, "mpm friction");
DEFINE_double(ppc, 1, "mpm ppc");
DEFINE_double(shift, 0.98, "shift");
DEFINE_double(damping, 10.0, "larger, more damping");

DEFINE_double(iiwa_link_0_x, 0.0, "iiwa base x coord");
DEFINE_double(iiwa_link_0_y, 0.0, "iiwa base y coord");
DEFINE_double(iiwa_link_0_z, 0.0, "iiwa base z coord");
DEFINE_double(iiwa_joint_1, 0.0, "iiwa joint 1");
DEFINE_double(iiwa_joint_2, 0.0, "iiwa joint 2");
DEFINE_double(iiwa_joint_3, 0.0, "iiwa joint 3");
DEFINE_double(iiwa_joint_4, 0.0, "iiwa joint 4");
DEFINE_double(iiwa_joint_5, 0.0, "iiwa joint 5");
DEFINE_double(iiwa_joint_6, 0.0, "iiwa joint 6");
DEFINE_double(iiwa_joint_7, 0.0, "iiwa joint 7");

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

// RigidTransformd FromXyzRpyDegree(const Vector3<double>& rpy_deg,
//                                  const Vector3<double>& p) {
//   return RigidTransformd(
//       math::RollPitchYaw<double>(rpy_deg * 3.1415926 / 180.0), p);
// }

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
    // fake update:
    VectorX<double> dX = current_state_values;
    dX.setZero();
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

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  MultibodyPlant<double> iiwa_controller_plant =
      MultibodyPlant<double>(plant_config.time_step);

  plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());
  multibody::Parser parser(&plant);
  const std::string filename = FindResourceOrThrow(
      "drake/manipulation/models/"
      "iiwa_description/iiwa7/iiwa7_no_collision.sdf");
  std::vector<drake::multibody::ModelInstanceIndex> instances =
      parser.AddModels(filename);
  auto iiwa = instances[0];
  Parser(&iiwa_controller_plant).AddModels(filename);

  RigidTransformd iiwa_position(Eigen::Vector3d(
      FLAGS_iiwa_link_0_x, FLAGS_iiwa_link_0_y, FLAGS_iiwa_link_0_z));

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("iiwa_link_0").body_frame(),
                   iiwa_position);
  iiwa_controller_plant.WeldFrames(
      iiwa_controller_plant.world_frame(),
      iiwa_controller_plant.GetBodyByName("iiwa_link_0").body_frame(),
      iiwa_position);

  // mpm stuff
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set =
          std::make_unique<drake::multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>(0.1, 0.1, 0.1));
  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model = std::make_unique<drake::multibody::mpm::constitutive_model::
                                   LinearCorotatedModel<double>>(1e5, 0.2);
  Vector3<double> translation = {10.0, 0.0, 0.5};
  std::unique_ptr<math::RigidTransform<double>> pose =
      std::make_unique<math::RigidTransform<double>>(translation);
  double h = 1.0;
  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set),
                                          std::move(model), std::move(pose),
                                          1000.0, h);
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  double Kp = 1000.0;
  double Kd = 2 * std::sqrt(Kp);

  drake::multibody::PdControllerGains gain(Kp, Kd);
  for (int i = 0; i < plant.num_actuators(); ++i) {
    plant.get_mutable_joint_actuator(drake::multibody::JointActuatorIndex(i))
        .set_controller_gains(gain);
  }

  plant.Finalize();
  iiwa_controller_plant.Finalize();

  Eigen::VectorXd iiwa_initial_joint_values(7);
  iiwa_initial_joint_values << FLAGS_iiwa_joint_1, FLAGS_iiwa_joint_2,
      FLAGS_iiwa_joint_3, FLAGS_iiwa_joint_4, FLAGS_iiwa_joint_5,
      FLAGS_iiwa_joint_6, FLAGS_iiwa_joint_7;

  // iiwa controller
  // first find the initial pose for the current joint values
  std::unique_ptr<Context<double>> temp_context = plant.CreateDefaultContext();
  plant.SetPositions(temp_context.get(), iiwa, iiwa_initial_joint_values);
  RigidTransformd iiwa_controller_initial_pose =
      plant.GetBodyByName("iiwa_link_7")
          .body_frame()
          .CalcPoseInWorld(*(temp_context.get()));
  fmt::print("iiwa_controller_initial_pose translation = {}\n",
               fmt_eigen(iiwa_controller_initial_pose.translation().transpose()));

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

  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.publish_period = 1.0 / 100;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(
      &builder, scene_graph, nullptr, visualize_params);
  // connect mpm to output port
  builder.Connect(deformable_model->mpm_particle_positions_port(),
                  visualizer.mpm_data_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);
  //   auto& diff_ik_context =
  //       diff_ik->GetMyMutableContextFromRoot(&mutable_context);

  plant.SetPositions(&plant_context, iiwa, iiwa_initial_joint_values);

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

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
