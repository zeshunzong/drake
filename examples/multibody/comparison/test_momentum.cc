#include <math.h>

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_point_cloud_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_with_plasticity.h"
#include "drake/multibody/mpm/constitutive_model/stvk_hencky_with_von_mises_model.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 2.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 3e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3,
              "Mass density of the deformable body [kg/m³]. We observe that "
              "density above 2400 kg/m³ makes the torus too heavy to be picked "
              "up by the suction gripper.");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(hydro_modulus, 1e8, "Hydroelastic modulus [Pa].");
DEFINE_double(damping, 1e2, "H&C damping.");
DEFINE_double(ppc, 50, "mpm ppc");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Capsule;
using drake::geometry::Ellipsoid;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  plant_config.discrete_contact_approximation = "lagged";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);


  ProximityProperties compliant_hydro_props;
  ProximityProperties rigid_hydro_props;

  const CoulombFriction<double> surface_friction(10.0, 10.0);
  AddContactMaterial(FLAGS_damping, {}, surface_friction,
                     &compliant_hydro_props);
  AddContactMaterial(FLAGS_damping, {}, surface_friction, &rigid_hydro_props);

  AddCompliantHydroelasticProperties(0.01, FLAGS_hydro_modulus,
                                     &compliant_hydro_props);
  AddRigidHydroelasticProperties(0.01, &rigid_hydro_props);
  /* Set up a ground. */
  Box ground{10, 10, 10};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -5});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_hydro_props);

  double rho1 = 2000; unused(rho1);
  double box_width = 0.3;
  const SpatialInertia<double> box1spatial =
      SpatialInertia<double>::SolidBoxWithDensity(rho1, box_width, box_width,
                                                  box_width);
  const RigidBody<double>& box1 = plant.AddRigidBody("box1", box1spatial);
  const Vector4<double> light_blue(0.5, 0.8, 1.0, 1.0);
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> green(0.0, 1.0, 0.0, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  const Vector4<double> dark_blue(0.0, 0.0, 0.8, 1.0);
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.2);
  unused(light_blue);
  unused(red);
  unused(green);
  unused(blue);
  unused(dark_blue);
  unused(orange);
  plant.RegisterVisualGeometry(box1, RigidTransformd::Identity(),
                               Box(box_width, box_width, box_width), "Cube1V",
                               light_blue);
  plant.RegisterCollisionGeometry(box1, RigidTransformd::Identity(),
                                  Box(box_width, box_width, box_width), "Cube1",
                                  compliant_hydro_props);
  /* Set up a deformable torus. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  // set a MPM body
  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set =
          std::make_unique<drake::multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>(box_width / 2.0, box_width / 2.0,
                              box_width / 2.0));

  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model = std::make_unique<drake::multibody::mpm::constitutive_model::
                                   LinearCorotatedModel<double>>(1e5, 0.2);

  std::unique_ptr<math::RigidTransform<double>> pose =
      std::make_unique<math::RigidTransform<double>>(
          Vector3<double>(0.0, 0.0, box_width / 2.0 + 0.0 * box_width));

  double h = 0.15;

  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set),
                                          std::move(model), std::move(pose),
                                          500.0, h);

  owned_deformable_model->SetMpmDamping(0.001);
  owned_deformable_model->SetMpmStiffness(1e6);
  owned_deformable_model->SetMpmMinParticlesPerCell(
      static_cast<int>(FLAGS_ppc));

  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

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

  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.publish_period = FLAGS_time_step * 1;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, meshcat_params);
  auto meshcat_pc_visualizer =
      builder.AddSystem<drake::geometry::MeshcatPointCloudVisualizer>(
          meshcat, "cloud", meshcat_params.publish_period);
  meshcat_pc_visualizer->set_point_size(0.005);
  builder.Connect(deformable_model->mpm_point_cloud_port(),
                  meshcat_pc_visualizer->cloud_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);
  unused(plant_context);
//   unused(plant_context);
  const double base_height = 0.15;
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("box1"),
                        math::RigidTransformd{Vector3d(0.0, 0, base_height+0.5)});

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  sleep(5);
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
      "A parallel (or suction) gripper grasps a deformable torus on the "
      "ground, lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
