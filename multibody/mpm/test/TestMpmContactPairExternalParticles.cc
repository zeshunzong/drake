#include "drake/multibody/mpm/BSpline.h"
#include "drake/multibody/mpm/SparseGrid.h"
#include "drake/multibody/mpm/MPMTransfer.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/CorotatedElasticModel.h"
#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"
#include <gtest/gtest.h>
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
    // this part does not really matter. We will create a separate particles
    

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


    // the whole point above is to get deformable_driver, nothing else is meaningful


    // consider a real particles object to be tested
    mpm::Particles<double> particles(0);
    SparseGrid<double> grid{0.2};
    MPMTransfer<double> mpm_transfer{};


    // add particle #1
    Eigen::Matrix3<double> F_in;
    F_in.setIdentity(); 
    CorotatedElasticModel<double> model(1.0,0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<double>> elastoplastic_model_p = model.Clone();
    particles.AddParticle(Eigen::Vector3<double>{0.1,-0.37,0.22}, Eigen::Vector3<double>{0,0,0}, 1.0, 1.0,
            F_in, Eigen::Matrix3<double>::Identity(), 
            Eigen::Matrix3<double>::Identity(),Eigen::Matrix3<double>::Zero(), std::move(elastoplastic_model_p));

    // add particle #2
    Eigen::Matrix3<double> F_in2;
    F_in2.setIdentity(); 
    CorotatedElasticModel<double> model2(1.0,0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<double>> elastoplastic_model_p2 = model2.Clone();
    particles.AddParticle(Eigen::Vector3<double>{0.07,-0.07,-0.12}, Eigen::Vector3<double>{0,0,0}, 1.0, 1.0,
            F_in2, Eigen::Matrix3<double>::Identity(), 
            Eigen::Matrix3<double>::Identity(),Eigen::Matrix3<double>::Zero(), std::move(elastoplastic_model_p2));

    // add particle #3
    Eigen::Matrix3<double> F_in3;
    F_in3.setIdentity();
    CorotatedElasticModel<double> model3(1.0, 0.3);
    std::unique_ptr<mpm::ElastoPlasticModel<double>> elastoplastic_model_p3 = model3.Clone();
    particles.AddParticle(Eigen::Vector3<double>{0.2,-0.4,0.25}, Eigen::Vector3<double>{0,0,0}, 1.0, 1.0,
            F_in3, Eigen::Matrix3<double>::Identity(), 
            Eigen::Matrix3<double>::Identity(),Eigen::Matrix3<double>::Zero(), std::move(elastoplastic_model_p3));

    particles.print_info();
    /* before ordering
    particle 0 position 0.1 -0.37 0.22
    particle 1 position 0.07 -0.07 -0.12
    particle 2 position 0.2 -0.4 0.25
    */

    int num_active_grids = mpm_transfer.MakeGridCompatibleWithParticles(&particles, &grid);

    particles.print_info();
    /* after ordering
    particle 0 position 0.07 -0.07 -0.12
    particle 1 position 0.1 -0.37 0.22
    particle 2 position 0.2 -0.4 0.25
    */

    const geometry::QueryObject<double>& query_object = discrete_update_manager.plant().get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<double>>(plant_context);
    
    // deformable_driver.DummyCheckContext(plant_context);

    std::vector<DiscreteContactPair<double>> result{};

    // in reality, what is really being called is AppendDiscreteContactPairsMpm(mbp's context, &result)
    // here we assume that the particles will be the same as evaluating from mbp's context
    deformable_driver.AppendDiscreteContactPairsMpm(query_object, particles, &result);

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
