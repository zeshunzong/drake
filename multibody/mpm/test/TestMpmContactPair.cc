#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/SparseGrid.h"
#include "drake/multibody/mpm/MPMTransfer.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"
#include <gtest/gtest.h>
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include <iostream>
#include <vector>
#include "drake/multibody/plant/deformable_model.h"
#include <stdlib.h> 
#include "drake/common/eigen_types.h"
#include <vector>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/plant/compliant_contact_manager.h"


namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

// constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();

using Eigen::Matrix3d;
using Eigen::Matrix3Xd;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::fem::DeformableBodyConfig;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using drake::geometry::GeometryId;
using drake::multibody::internal::DiscreteUpdateManager;
using drake::multibody::internal::CompliantContactManager;
using drake::multibody::internal::DiscreteContactPair;


// Omit this file!!!
// Omit this file!!!
// Omit this file!!!
// Omit this file!!!
// Omit this file!!!
// Maybe recover this if we see inconsistency between particles evaluated from context and external particles
void CreateScene(){
    bool dynamic_rigid_body = false;
    
    GeometryId rigid_geometry_id_;
    BodyIndex rigid_body_index_;
    systems::DiagramBuilder<double> builder;
    MultibodyPlantConfig plant_config;
    plant_config.discrete_contact_solver = "sap";
    auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

    Box rigid_box{4, 4, 4};
    const RigidTransformd X_WR(Eigen::Vector3d{0, 0, -2});

    ProximityProperties rigid_proximity_props;
    geometry::AddContactMaterial({}, {}, CoulombFriction<double>(1.0, 1.0),
                                 &rigid_proximity_props);
    rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                      geometry::internal::kRezHint, 1.0);
    if (dynamic_rigid_body) {
        const RigidBody<double>& rigid_body =
          plant.AddRigidBody("rigid_body", SpatialInertia<double>());

        rigid_geometry_id_ = plant.RegisterCollisionGeometry(
          rigid_body, X_WR, rigid_box,
          "dynamic_collision_geometry", rigid_proximity_props);

        rigid_body_index_ = rigid_body.index();
    } else {

        /* Register the same rigid geometry, but static instead of dynamic. */
        rigid_geometry_id_ = plant.RegisterCollisionGeometry(
            plant.world_body(), X_WR, rigid_box,
            "static_collision_geometry", rigid_proximity_props);

        rigid_body_index_ = plant.world_body().index();
        
    }

    /* register a mpm body here */
    auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

    double E = 5e4;
    double nu = 0.4;
    std::unique_ptr<multibody::mpm::ElastoPlasticModel<double>> constitutive_model
            = std::make_unique<multibody::mpm::CorotatedElasticModel<double>>(E, nu);

    multibody::SpatialVelocity<double> geometry_initial_veolocity;
    geometry_initial_veolocity.translational() = Vector3<double>{0.0, 0.0, 0.0};//{0.1, 0.1, 0.1};
    geometry_initial_veolocity.rotational() = Vector3<double>{0.0, 0.0, 0.0};//{M_PI/2, M_PI/2, M_PI/2};

    Vector3<double> geometry_translation = {0.0, 0.0, 0.05};
    math::RigidTransform<double> geometry_pose = math::RigidTransform<double>(geometry_translation);

    double density = 1000.0; double grid_h = 0.2;
    int min_num_particles_per_cell = 1;

    std::unique_ptr<multibody::mpm::AnalyticLevelSet<double>> mpm_geometry_level_set = 
                                    std::make_unique<multibody::mpm::SphereLevelSet<double>>(0.2);

    owned_deformable_model->RegisterMpmBody(
      std::move(mpm_geometry_level_set), std::move(constitutive_model), geometry_initial_veolocity,
      geometry_pose, density, grid_h, min_num_particles_per_cell);

    /* 
    Four particles should be below z=0
    particle #1 posiion: -0.113647 -0.0631355 -0.051133
    particle #10 posiion: -0.011385 0.0588435 -0.074056
    particle #15 posiion: 0.0799938 -0.090995 -0.0950756
    particle #17 posiion: -0.00799091 -0.0601236 -0.0492368
    */ 

    const DeformableModel<double>* deformable_model = owned_deformable_model.get();
    plant.AddPhysicalModel(std::move(owned_deformable_model));
    plant.Finalize();



    builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));
    builder.Connect(
        deformable_model->mpm_particle_positions_port(),
        scene_graph.mpm_data_input_port());

    auto diagram = builder.Build();

    auto diagram_context = diagram->CreateDefaultContext();
    const Context<double>& plant_context = plant.GetMyContextFromRoot(*diagram_context);
    const multibody::internal::DiscreteUpdateManager<double>& discrete_update_manager = plant.GetDiscreteUpdateManager();
    const multibody::internal::CompliantContactManager<double>& compliant_contact_manager = reinterpret_cast<const CompliantContactManager<double>&>(discrete_update_manager);
    const multibody::internal::DeformableDriver<double>& deformable_driver = compliant_contact_manager.GetDeformableDriver();
    
    deformable_driver.DummyCheckContext(plant_context);

    std::vector<DiscreteContactPair<double>> result{};

    deformable_driver.AppendDiscreteContactPairsMpm(plant_context, &result);

    std::cout << "ssss " << result.size() << std::endl;
    std::cout << result[0].p_WC[0] << " " <<result[0].p_WC[1] << " " << result[0].p_WC[2] <<std::endl;
    std::cout << result[0].id_A <<std::endl;
    std::cout << result[0].id_B <<std::endl;
    std::cout << result[0].phi0 <<std::endl;
    

    EXPECT_TRUE(false);
}



GTEST_TEST(TestMpmContactPair, testcontact) {
    // TestEnergyAndForceAndHessian();
    CreateScene();

    
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
