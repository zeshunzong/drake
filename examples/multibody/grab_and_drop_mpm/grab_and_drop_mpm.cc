#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 3.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(friction, 0.0, "mpm friction");
DEFINE_double(ppc, 8, "mpm ppc");
DEFINE_double(shift, 0.98, "shift");
DEFINE_double(damping, 10.0, "larger, more damping");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

/* We create a leaf system that uses PD control to output a force signal to
 a gripper to follow a close-lift-open motion sequence. The signal is
 2-dimensional with the first element corresponding to the wrist degree of
 freedom and the second element corresponding to the left finger degree of
 freedom. This control is a time-based state machine, where forces change based
 on the context time. This is strictly for demo purposes and is not intended to
 generalize to other cases. There are four states: 0. The fingers are open in
 the initial state.
  1. The fingers are closed to secure a grasp.
  2. The gripper is lifted to a prescribed final height.
  3. The fingers are open to loosen a grasp.
 The desired state is interpolated between these states. */
class GripperPositionControl : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperPositionControl);

  /* Constructs a GripperPositionControl system with the given parameters.
   @param[in] open_width   The width between fingers in the open state. (meters)
   @param[in] closed_width The width between fingers in the closed state.
                           (meters)
   @param[in] height       The height of the gripper in the lifted state.
                           (meters) */
  GripperPositionControl(double open_width, double closed_width, double height)
      : initial_state_(0, -open_width / 2),
        closed_state_(0, -closed_width / 2),
        lifted_state_(height, -closed_width / 2),  // lifted and closed
        open_state_(height, -open_width / 2)       // lifted andopen
  {
    this->DeclareVectorOutputPort("gripper force", BasicVector<double>(2),
                                  &GripperPositionControl::SetAppliedForce);
    this->DeclareVectorInputPort("gripper state", BasicVector<double>(6));
  }

 private:
  void SetAppliedForce(const Context<double>& context,
                       BasicVector<double>* output) const {
    const VectorXd gripper_state =
        EvalVectorInput(context, GetInputPort("gripper state").get_index())
            ->get_value();
    /* There are 6 dofs in the state, corresponding to
     q_translate_joint, q_left_finger, q_right_finger,
     v_translate_joint, v_left_finger, v_right_finger. */
    /* The positions of the translate joint and the left finger. */
    const Vector2d measured_positions = gripper_state.head(2);
    /* The velocities of the translate joint and the left finger. */
    const Vector2d measured_velocities = gripper_state.segment<2>(3);
    const Vector2d desired_velocities(0, 0);
    Vector2d desired_positions;
    const double t = context.get_time();

    if (t < fingers_wait_time_) {
      desired_positions = initial_state_;
    } else if (t < fingers_closed_time_) {
      const double end_time = fingers_closed_time_;
      const double theta = t / end_time;
      desired_positions =
          theta * closed_state_ + (1.0 - theta) * initial_state_;
    } else if (t < gripper_lifted_time_) {
      const double end_time = gripper_lifted_time_ - fingers_closed_time_;
      const double theta = (t - fingers_closed_time_) / end_time;
      desired_positions = theta * lifted_state_ + (1.0 - theta) * closed_state_;
    } else if (t < hold_time_) {
      desired_positions = lifted_state_;
    } else if (t < fingers_open_time_) {
      const double end_time = fingers_open_time_ - hold_time_;
      const double theta = (t - hold_time_) / end_time;
      desired_positions = theta * open_state_ + (1.0 - theta) * lifted_state_;
    } else {
      desired_positions = open_state_;
    }
    const Vector2d force = kp_ * (desired_positions - measured_positions) +
                           kd_ * (desired_velocities - measured_velocities);

    output->get_mutable_value() << force;
  }

  const double fingers_wait_time_{1.0};
  /* The time at which the fingers reach the desired closed state. */
  const double fingers_closed_time_{1.4};
  /* The time at which the gripper reaches the desired "lifted" state. */
  const double gripper_lifted_time_{1.8};
  const double hold_time_{1.9};
  /* The time at which the fingers reach the desired open state. */
  const double fingers_open_time_{2.1};
  Vector2d initial_state_;
  Vector2d closed_state_;
  Vector2d lifted_state_;
  Vector2d open_state_;
  const double kp_{1400};
  const double kd_{60.0};
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_approximation = "lagged";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tessellated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.15, 1.15);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 1.0);
  /* Set up a ground. */
  Box ground{10, 10, 10};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -5.00});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", std::move(illustration_props));

  /* Set up a simple gripper. */
  Parser parser(&plant);
  parser.AddModelsFromUrl(
      "package://drake/examples/multibody/grab_and_drop_mpm/"
      "simple_gripper.sdf");
  /* Add collision geometries. */
  const RotationMatrix<double> Rot = RotationMatrix<double>::MakeFromOneVector(
      Eigen::Vector3d{0.0, 0.0, 1.0}, 1);

  const RigidTransformd X_BG(Rot, Eigen::Vector3d{0.0, -0.01, 1.0 - 0.9});
  const Body<double>& left_finger = plant.GetBodyByName("left_finger");
  const Body<double>& right_finger = plant.GetBodyByName("right_finger");
  /* The size of the fingers is set to match the visual geometries in
   simple_gripper.sdf. */
  plant.RegisterCollisionGeometry(left_finger, X_BG, Box(0.007, 0.081, 0.028),
                                  "left_finger_collision",
                                  rigid_proximity_props);
  plant.RegisterCollisionGeometry(right_finger, X_BG, Box(0.007, 0.081, 0.028),
                                  "left_finger_collision",
                                  rigid_proximity_props);
  IllustrationProperties illustration_props2;
  illustration_props2.AddProperty("phong", "diffuse",
                                  Vector4d(1.0, 1.0, 1.0, 1.0));
  plant.RegisterVisualGeometry(left_finger, X_BG, Box(0.007, 0.081, 0.028),
                               "left_finger_visual", illustration_props2);
  plant.RegisterVisualGeometry(right_finger, X_BG, Box(0.007, 0.081, 0.028),
                               "right_finger_visual", illustration_props2);

  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  double width = 0.036;
  std::unique_ptr<multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set =
          std::make_unique<multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>{width * 1.5, width, width});
  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model = std::make_unique<drake::multibody::mpm::constitutive_model::
                                   LinearCorotatedModel<double>>(1e5, 0.4);
  multibody::SpatialVelocity<double> geometry_initial_veolocity;
  geometry_initial_veolocity.translational() =
      Vector3<double>{0.0, 0.0, 0.0};  //{0.1, 0.1, 0.1};
  geometry_initial_veolocity.rotational() =
      Vector3<double>{0.0, 0.0, 0.0};  //{M_PI/2, M_PI/2, M_PI/2};
  Vector3<double> translation = {-0.97 + 0.9, 0.062, width + 0};
  std::unique_ptr<math::RigidTransform<double>> pose =
      std::make_unique<math::RigidTransform<double>>(translation);
  double h = width / 2;
  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set),
                                          std::move(model), std::move(pose),
                                          1000.0, h);

  const Vector3<double> mpm_gravity{0.0, 0.0, -2};
  owned_deformable_model->SetMpmGravity(mpm_gravity);

  owned_deformable_model->SetMpmFriction(FLAGS_friction);
  owned_deformable_model->SetMpmDamping(FLAGS_damping);
  owned_deformable_model->SetMpmMinParticlesPerCell(
      static_cast<int>(FLAGS_ppc));
  // owned_deformable_model->ApplyMpmGround();

  const double scale = 0.65;
  // /* Minor diameter of the torus inferred from the vtk file. */
  const double kL = 0.09 * scale;

  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* Get joints so that we can set constraints and initial conditions. */
  PrismaticJoint<double>& left_slider =
      plant.GetMutableJointByName<PrismaticJoint>("left_slider");
  PrismaticJoint<double>& right_slider =
      plant.GetMutableJointByName<PrismaticJoint>("right_slider");
  /* Constrain the left and the right fingers such that qₗ = -qᵣ. */
  plant.AddCouplerConstraint(left_slider, right_slider, -1.0);
  /* Viscous damping for the finger joints, in N⋅s/m. */
  left_slider.set_default_damping(50.0);
  right_slider.set_default_damping(50.0);

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.publish_period = 1.0 / 64;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(
      &builder, scene_graph, nullptr, params);
  // connect mpm to output port
  builder.Connect(deformable_model->mpm_particle_positions_port(),
                  visualizer.mpm_data_input_port());

  /* Set the width between the fingers for open and closed states as well as the
   height to which the gripper lifts the deformable torus. */
  const double open_width = kL * 1.5;
  const double closed_width = kL * 0.5;
  const double lifted_height = 0.12;

  const auto& control = *builder.AddSystem<GripperPositionControl>(
      open_width, closed_width, lifted_height);
  builder.Connect(plant.get_state_output_port(), control.get_input_port());
  builder.Connect(control.get_output_port(), plant.get_actuation_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Set initial conditions for the gripper. */
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  left_slider.set_translation(&plant_context, -open_width / 2.0);
  right_slider.set_translation(&plant_context, open_width / 2.0);

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
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
