#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/temp_directory.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_animation.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/rgba.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/meshcat/contact_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"

/* To test, you must manually run `bazel run //geometry:meshcat_manual_test`,
then follow the instructions on your console. */

namespace drake {
namespace geometry {

using Eigen::Vector3d;
using math::RigidTransformd;
using math::RotationMatrixd;

int do_main() {
  auto meshcat = std::make_shared<Meshcat>();

  // For every two items we add to the initial array, decrement start_x by one
  // to keep things centered.
  // Use ++x as the x-position of new items.
  const double start_x = -1;
  double x = start_x;


  {
    const int kPoints = 100;
    perception::PointCloud cloud(
        kPoints, perception::pc_flags::kXYZs | perception::pc_flags::kRGBs);
    Eigen::Matrix3Xf m = Eigen::Matrix3Xf::Random(3, kPoints);
    cloud.mutable_xyzs() = Eigen::DiagonalMatrix<float, 3>{0.05, 0.05, 0.1} * m;
    cloud.mutable_rgbs() = (255.0 * (m.array() + 1.0) / 2.0).cast<uint8_t>(); // if do not do this, color is black

    // up to here we get a pointCloud with n points

    meshcat->SetObject("point_cloud", cloud, 0.01);
    meshcat->SetTransform("point_cloud", RigidTransformd(Vector3d{++x, 0, 0}));
  }



  std::cout << R"""(
Open up your browser to the URL above.

- The background should be grey.
- From left to right along the x axis, you should see:
  - a red sphere
  - a green cylinder (with the long axis in z)
  - a pink semi-transparent ellipsoid (long axis in z)
  - a blue box (long axis in z)
  - a teal capsule (long axis in z)
  - a red cone (expanding in +z, twice as wide in y than in x)
  - a bright green cube (the green comes from a texture map)
  - a yellow mustard bottle w/ label
  - a dense rainbow point cloud in a box (long axis in z)
  - a blue line coiling up (in z).
  - 4 green vertical line segments (in z).
  - a purple triangle mesh with 2 faces.
  - the same purple triangle mesh drawn as a wireframe.
  - the same triangle mesh drawn in multicolor.
  - a blue mesh plot of the function z = y*sin(5*x).
)""";
  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


 
  std::cout
      << "- The blue box should have disappeared\n"
      << "- The lights should have dimmed.\n"
      << "- The background should have been disabled (it will appear white)"
      << std::endl;
  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  meshcat->Delete();
  std::cout << "- Everything else should have disappeared." << std::endl;

  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  meshcat->SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.6);

  {
    systems::DiagramBuilder<double> builder;
    auto [plant, scene_graph] =
        multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);

    multibody::Parser parser(&plant);

    // Add the hydroelastic spheres and joints between them.
    const std::string hydro_sdf =
        FindResourceOrThrow("drake/multibody/meshcat/test/hydroelastic.sdf");
    parser.AddModels(hydro_sdf);
    const auto& body1 = plant.GetBodyByName("body1");
    plant.AddJoint<multibody::PrismaticJoint>("body1", plant.world_body(),
                                              std::nullopt, body1, std::nullopt,
                                              Vector3d::UnitZ());
    const auto& body2 = plant.GetBodyByName("body2");
    plant.AddJoint<multibody::PrismaticJoint>("body2", plant.world_body(),
                                              std::nullopt, body2, std::nullopt,
                                              Vector3d::UnitX());
    plant.Finalize();

    MeshcatVisualizerParams params;
    params.delete_on_initialization_event = false;
    auto& visualizer = MeshcatVisualizerd::AddToBuilder(
        &builder, scene_graph, meshcat, std::move(params));

    multibody::meshcat::ContactVisualizerParams cparams;
    cparams.newtons_per_meter = 60.0;
    auto& contact = multibody::meshcat::ContactVisualizerd::AddToBuilder(
        &builder, plant, meshcat, std::move(cparams));

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();

    plant.SetPositions(&plant.GetMyMutableContextFromRoot(context.get()),
                       Eigen::Vector2d{0.1, 0.3});
    diagram->ForcedPublish(*context);
    std::cout << "- Now you should see three colliding hydroelastic spheres."
              << std::endl;
    std::cout << "[Press RETURN to continue]." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    contact.Delete();
    visualizer.Delete();
  }

  {
    systems::DiagramBuilder<double> builder;
    auto [plant, scene_graph] =
        multibody::AddMultibodyPlantSceneGraph(&builder, 0.001);
    multibody::Parser parser(&plant);
    parser.AddModels(
        FindResourceOrThrow("drake/manipulation/models/iiwa_description/urdf/"
                            "iiwa14_spheres_collision.urdf"));
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"));
    parser.AddModels(FindResourceOrThrow(
        "drake/examples/kuka_iiwa_arm/models/table/"
        "extra_heavy_duty_table_surface_only_collision.sdf"));
    const double table_height = 0.7645;
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("link"),
                     RigidTransformd(Vector3d{0, 0, -table_height - 0.01}));
    plant.Finalize();

    builder.ExportInput(plant.get_actuation_input_port(), "actuation_input");
    MeshcatVisualizerParams params;
    params.delete_on_initialization_event = false;
    auto& visualizer = MeshcatVisualizerd::AddToBuilder(
        &builder, scene_graph, meshcat, std::move(params));

    multibody::meshcat::ContactVisualizerParams cparams;
    cparams.newtons_per_meter = 60.0;
    multibody::meshcat::ContactVisualizerd::AddToBuilder(
        &builder, plant, meshcat, std::move(cparams));

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();
    diagram->get_input_port().FixValue(context.get(), Eigen::VectorXd::Zero(7));

    diagram->ForcedPublish(*context);
    std::cout
        << "- Now you should see a kuka model (from MultibodyPlant/SceneGraph)"
        << std::endl;

    std::cout << "[Press RETURN to continue]." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << "Now we'll run the simulation...\n"
              << "- You should see the robot fall down and hit the table\n"
              << "- You should see the contact force vectors (when it hits)\n"
              << std::endl;

    systems::Simulator<double> simulator(*diagram, std::move(context));
    simulator.set_target_realtime_rate(1.0);
    visualizer.StartRecording();
    simulator.AdvanceTo(4.0);
    visualizer.PublishRecording();

    std::cout
        << "The recorded simulation results should now be available as an "
           "animation.  Use the animation GUI to confirm."
        << std::endl;

    std::cout << "[Press RETURN to continue]." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  const std::string html_filename(temp_directory() + "/meshcat_static.html");
  std::ofstream html_file(html_filename);
  html_file << meshcat->StaticHtml();
  html_file.close();

  std::cout << "A standalone HTML file capturing this scene (including the "
               "animation) has been written to file://"
            << html_filename
            << "\nOpen that location in your browser now and confirm that "
               "the iiwa is visible and the animation plays."
            << std::endl;

  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::remove(html_filename.c_str());
  std::cout
      << "Note: I've deleted the temporary HTML file (it's several Mb).\n\n";

  meshcat->AddButton("ButtonTest");
  meshcat->AddButton("Press t Key");
  meshcat->AddButton("Press t Key", "KeyT");  // Now the keycode is assigned.
  meshcat->AddSlider("SliderTest", 0, 1, 0.01, 0.5, "ArrowLeft", "ArrowRight");

  std::cout << "I've added two buttons and a slider to the controls menu.\n";
  std::cout << "- Click the ButtonTest button a few times.\n";
  std::cout << "- Press the 't' key in the meshcat window, which "
               "should be equivalent to pressing the second button.\n";
  std::cout << "The buttons do nothing, but the total number of clicks for "
               "each button will be reported after you press RETURN.\n";
  std::cout << "- Move SliderTest slider.\n";
  std::cout << "- Confirm that the ArrowLeft and ArrowRight keys also move the "
               "slider.\n";
  std::cout << "- Open a second browser (" << meshcat->web_url()
            << ") and confirm that moving the slider in one updates the slider "
               "in the other.\n";

  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::cout << "Got " << meshcat->GetButtonClicks("ButtonTest")
            << " clicks on ButtonTest.\n"
            << "Got " << meshcat->GetButtonClicks("Press t Key")
            << " clicks on \"Press t Key\".\n"
            << "Got " << meshcat->GetSliderValue("SliderTest")
            << " value for SliderTest.\n\n" << std::endl;

  std::cout << "Next, we'll test gamepad (i.e., joystick) features.\n\n";
  std::cout
      << "While the Meshcat browser window has focus, click any button on "
      << "your gamepad to activate gamepad support in the browser.\n\n";
  std::cout
      << "Then(after you press RETURN), we'll print the gamepad stats for 5 "
      << "seconds. During that time, move the control sticks and hold some "
      << "buttons and you should see those values reflected in the printouts. "
      << "As long as you see varying values as you move the controls, that's "
      << "sufficient to consider the test passing; the exact values do not "
      << "matter.\n";

  std::cout << "[Press RETURN to continue]." << std::endl;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  Meshcat::Gamepad gamepad = meshcat->GetGamepad();
  if (!gamepad.index) {
    std::cout << "No gamepad activity detected.\n";
  } else {
    for (int i = 0; i < 5; ++i) {
      gamepad = meshcat->GetGamepad();
      std::cout << "Gamepad status:\n";
      std::cout << "  gamepad index: " << *gamepad.index << "\n";
      std::cout << "  buttons: ";
      for (auto const& value : gamepad.button_values) {
        std::cout << value << ", ";
      }
      std::cout << "\n";
      std::cout << "  axes: ";
      for (auto const& value : gamepad.axes) {
        std::cout << value << ", ";
      }
      std::cout << "\n";
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  std::cout << "Exiting..." << std::endl;
  return 0;
}

}  // namespace geometry
}  // namespace drake

int main() { return drake::geometry::do_main(); }
