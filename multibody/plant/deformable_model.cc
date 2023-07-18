#include "drake/multibody/plant/deformable_model.h"

#include <algorithm>
#include <utility>

#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/fem_state.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_corotated_model.h"
#include "drake/multibody/fem/linear_simplex_element.h"
#include "drake/multibody/fem/simplex_gaussian_quadrature.h"
#include "drake/multibody/fem/volumetric_model.h"


namespace drake {
namespace multibody {

using geometry::FrameId;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::SceneGraph;
using geometry::SourceId;

using fem::DeformableBodyConfig;
using fem::MaterialModel;

// using drake::multibody::fem::FemModel;
// using drake::multibody::fem::FemState;
// using drake::multibody::mpm::MpmState;
// using drake::multibody::mpm::MpmModel;


template <typename T>
DeformableBodyId DeformableModel<T>::RegisterDeformableBody(
    std::unique_ptr<geometry::GeometryInstance> geometry_instance,
    const fem::DeformableBodyConfig<T>& config, double resolution_hint) {
  this->ThrowIfSystemResourcesDeclared(__func__);
  /* Register the geometry with SceneGraph. */
  SceneGraph<T>& scene_graph = this->mutable_scene_graph(plant_);
  SourceId source_id = plant_->get_source_id().value();
  /* All deformable bodies are registered with the world frame at the moment. */
  const FrameId world_frame_id = scene_graph.world_frame_id();
  GeometryId geometry_id = scene_graph.RegisterDeformableGeometry(
      source_id, world_frame_id, std::move(geometry_instance), resolution_hint);

  /* Record the reference positions. */
  const geometry::SceneGraphInspector<T>& inspector =
      scene_graph.model_inspector();
  const geometry::VolumeMesh<double>* mesh_G =
      inspector.GetReferenceMesh(geometry_id);
  DRAKE_DEMAND(mesh_G != nullptr);
  const math::RigidTransform<T>& X_WG = inspector.GetPoseInFrame(geometry_id);
  geometry::VolumeMesh<double> mesh_W = *mesh_G;
  mesh_W.TransformVertices(X_WG);
  VectorX<T> reference_position(3 * mesh_W.num_vertices());
  for (int v = 0; v < mesh_W.num_vertices(); ++v) {
    reference_position.template segment<3>(3 * v) = mesh_W.vertex(v);
  }

  const DeformableBodyId body_id = DeformableBodyId::get_new_id();
  /* Build FEM model for the deformable body. */
  BuildLinearVolumetricModel(body_id, mesh_W, config);

  /* Do the book-keeping. */
  reference_positions_.emplace(body_id, std::move(reference_position));
  body_id_to_geometry_id_.emplace(body_id, geometry_id);
  geometry_id_to_body_id_.emplace(geometry_id, body_id);
  body_ids_.emplace_back(body_id);
  return body_id;
}

template <typename T>
DeformableBodyId DeformableModel<T>::RegisterMpmBody(
    std::unique_ptr<geometry::GeometryInstance> geometry_instance,
    const fem::DeformableBodyConfig<T>& config, double resolution_hint) {
  this->ThrowIfSystemResourcesDeclared(__func__);
  std::cout << "Temporary:Inputs to RegisterMpmBody are not used." << std::endl; getchar();

  
  if (ExistsMpmModel()) {
    throw std::logic_error("we only allow one mpm model");
  }

  mpm_model_ = std::make_unique<mpm::MpmModel<T>>(); // should add physical parameters to mpm_model_

  // multibody::SpatialVelocity<double> velocity_sphere;
  //   velocity_sphere.translational() = Vector3<double>{0.1, 0.1, 0.1};
  //   velocity_sphere.rotational() = Vector3<double>{M_PI/2, M_PI/2, M_PI/2};

  //   mpm::SphereLevelSet level_set_sphere = mpm::SphereLevelSet(0.2);
  //   Vector3<double> translation_sphere = {0.0, 0.0, 0.0};
  //   math::RigidTransform<double> pose_sphere =
  //                           math::RigidTransform<double>(translation_sphere);

  //   double E = 5e4;
  //   double nu = 0.4;
  //   std::unique_ptr<mpm::CorotatedElasticModel> elastoplastic_model
  //           = std::make_unique<mpm::CorotatedElasticModel>(E, nu);
  //   typename mpm::MpmModel<T>::MaterialParameters m_param_sphere{
  //                                               std::move(elastoplastic_model),
  //                                               1000,
  //                                               velocity_sphere,
  //                                               1
  //                                               };

  mpm_model_->set_grid_h(0.025);
  // mpm_model_->set_spatial_velocity(velocity_sphere);
  // mpm_model_->set_level_set(level_set_sphere);
  // mpm_model_->set_pose(pose_sphere);
  // mpm_model_->set_material_params(m_param_sphere);
  
  const DeformableBodyId body_id = DeformableBodyId::get_new_id();
  return body_id;
}




template <typename T>
void DeformableModel<T>::SetWallBoundaryCondition(DeformableBodyId id,
                                                  const Vector3<T>& p_WQ,
                                                  const Vector3<T>& n_W) {
  this->ThrowIfSystemResourcesDeclared(__func__);
  ThrowUnlessRegistered(__func__, id);
  DRAKE_DEMAND(n_W.norm() > 1e-10);
  const Vector3<T>& nhat_W = n_W.normalized();

  fem::FemModel<T>& fem_model = *fem_models_.at(id);
  const int num_nodes = fem_model.num_nodes();
  constexpr int kDim = 3;
  auto is_inside_wall = [&p_WQ, &nhat_W](const Vector3<T>& p_WV) {
    T distance_to_wall = (p_WV - p_WQ).dot(nhat_W);
    return distance_to_wall < 0;
  };

  const VectorX<T>& p_WVs = GetReferencePositions(id);
  fem::internal::DirichletBoundaryCondition<T> bc;
  for (int n = 0; n < num_nodes; ++n) {
    const int dof_index = kDim * n;
    const auto p_WV = p_WVs.template segment<kDim>(dof_index);
    if (is_inside_wall(p_WV)) {
      /* Set this node to be subject to zero Dirichlet BC. */
      bc.AddBoundaryCondition(fem::FemNodeIndex(n),
                              {p_WV, Vector3<T>::Zero(), Vector3<T>::Zero()});
    }
  }
  fem_model.SetDirichletBoundaryCondition(std::move(bc));
}

template <typename T>
systems::DiscreteStateIndex DeformableModel<T>::GetDiscreteStateIndex(
    DeformableBodyId id) const {
  this->ThrowIfSystemResourcesNotDeclared(__func__);
  ThrowUnlessRegistered(__func__, id);
  return discrete_state_indexes_.at(id);
}

template <typename T>
const fem::FemModel<T>& DeformableModel<T>::GetFemModel(
    DeformableBodyId id) const {
  ThrowUnlessRegistered(__func__, id);
  return *fem_models_.at(id);
}

template <typename T>
const mpm::MpmModel<T>& DeformableModel<T>::GetMpmModel() const {
  if (mpm_model_== nullptr){
    throw std::logic_error("GetMpmModel(): No MPM Model registered");
  }
  return *mpm_model_;
}

template <typename T>
const VectorX<T>& DeformableModel<T>::GetReferencePositions(
    DeformableBodyId id) const {
  ThrowUnlessRegistered(__func__, id);
  return reference_positions_.at(id);
}

template <typename T>
DeformableBodyId DeformableModel<T>::GetBodyId(
    DeformableBodyIndex index) const {
  this->ThrowIfSystemResourcesNotDeclared(__func__);
  DRAKE_THROW_UNLESS(index.is_valid() && index < num_bodies());
  return body_ids_[index];
}

template <typename T>
DeformableBodyIndex DeformableModel<T>::GetBodyIndex(
    DeformableBodyId id) const {
  this->ThrowIfSystemResourcesNotDeclared(__func__);
  ThrowUnlessRegistered(__func__, id);
  return body_id_to_index_.at(id);
}

template <typename T>
GeometryId DeformableModel<T>::GetGeometryId(DeformableBodyId id) const {
  ThrowUnlessRegistered(__func__, id);
  return body_id_to_geometry_id_.at(id);
}

template <typename T>
DeformableBodyId DeformableModel<T>::GetBodyId(
    geometry::GeometryId geometry_id) const {
  if (geometry_id_to_body_id_.count(geometry_id) == 0) {
    throw std::runtime_error(
        fmt::format("The given GeometryId {} does not correspond to a "
                    "deformable body registered with this model.",
                    geometry_id));
  }
  return geometry_id_to_body_id_.at(geometry_id);
}

template <typename T>
void DeformableModel<T>::BuildMpmModel(
    DeformableBodyId id,
    const fem::DeformableBodyConfig<T>& config,
    const mpm::Particles& particles) {

}

template <typename T>
void DeformableModel<T>::BuildLinearVolumetricModel(
    DeformableBodyId id, const geometry::VolumeMesh<double>& mesh,
    const fem::DeformableBodyConfig<T>& config) {
  if (fem_models_.find(id) != fem_models_.end()) {
    throw std::logic_error("An FEM model with id: " + to_string(id) +
                           " already exists.");
  }
  switch (config.material_model()) {
    case MaterialModel::kLinear:
      BuildLinearVolumetricModelHelper<fem::internal::LinearConstitutiveModel>(
          id, mesh, config);
      break;
    case MaterialModel::kCorotated:
      BuildLinearVolumetricModelHelper<fem::internal::CorotatedModel>(id, mesh,
                                                                      config);
      break;
    case MaterialModel::kLinearCorotated:
      BuildLinearVolumetricModelHelper<fem::internal::LinearCorotatedModel>(
          id, mesh, config);
      break;
  }
}

template <typename T>
template <template <typename, int> class Model>
void DeformableModel<T>::BuildLinearVolumetricModelHelper(
    DeformableBodyId id, const geometry::VolumeMesh<double>& mesh,
    const fem::DeformableBodyConfig<T>& config) {
  constexpr int kNaturalDimension = 3;
  constexpr int kSpatialDimension = 3;
  constexpr int kQuadratureOrder = 1;
  using QuadratureType =
      fem::internal::SimplexGaussianQuadrature<kNaturalDimension,
                                               kQuadratureOrder>;
  constexpr int kNumQuads = QuadratureType::num_quadrature_points;
  using IsoparametricElementType =
      fem::internal::LinearSimplexElement<T, kNaturalDimension,
                                          kSpatialDimension, kNumQuads>;
  using ConstitutiveModelType = Model<T, kNumQuads>;
  static_assert(
      std::is_base_of_v<
          fem::internal::ConstitutiveModel<
              ConstitutiveModelType, typename ConstitutiveModelType::Traits>,
          ConstitutiveModelType>,
      "The template parameter 'Model' must be derived from "
      "ConstitutiveModel.");
  using FemElementType =
      fem::internal::VolumetricElement<IsoparametricElementType, QuadratureType,
                                       ConstitutiveModelType>;
  using FemModelType = fem::internal::VolumetricModel<FemElementType>;

  const fem::DampingModel<T> damping_model(
      config.mass_damping_coefficient(),
      config.stiffness_damping_coefficient());

  auto fem_model = std::make_unique<FemModelType>();
  ConstitutiveModelType constitutive_model(config.youngs_modulus(),
                                           config.poissons_ratio());
  typename FemModelType::VolumetricBuilder builder(fem_model.get());
  builder.AddLinearTetrahedralElements(mesh, constitutive_model,
                                       config.mass_density(), damping_model);
  builder.Build();

  fem_models_.emplace(id, std::move(fem_model));
}

template <typename T>
void DeformableModel<T>::DoDeclareSystemResources(MultibodyPlant<T>* plant) {
  std::cout << "calling do declare system resources" << std::endl;
  /* Ensure that the owning plant is the one declaring system resources. */
  DRAKE_DEMAND(plant == plant_);
  /* Declare discrete states. */
  for (const auto& [deformable_id, fem_model] : fem_models_) {
    std::unique_ptr<fem::FemState<T>> default_fem_state =
        fem_model->MakeFemState();
    const int num_dofs = default_fem_state->num_dofs();
    VectorX<T> model_state(num_dofs * 3 /* q, v, and a */);
    model_state.head(num_dofs) = default_fem_state->GetPositions();
    model_state.segment(num_dofs, num_dofs) =
        default_fem_state->GetVelocities();
    model_state.tail(num_dofs) = default_fem_state->GetAccelerations();
    discrete_state_indexes_.emplace(
        deformable_id, this->DeclareDiscreteState(plant, model_state));
  }

  /* Declare the vertex position output port. */
  vertex_positions_port_index_ =
      this->DeclareAbstractOutputPort(
              plant, "vertex_positions",
              []() {
                return AbstractValue::Make<
                    geometry::GeometryConfigurationVector<T>>();
              },
              [this](const systems::Context<T>& context,
                     AbstractValue* output) {
                this->CopyVertexPositions(context, output);
              },
              {systems::System<double>::xd_ticket()})
          .get_index();


  // output port for visualization
    mpm_particle_positions_port_index_ =
        this->DeclareAbstractOutputPort(
                plant, "mpm",
                []() {
                  return AbstractValue::Make<
                      std::vector<Vector3<double>>>();
                },
                [this](const systems::Context<T>& context,
                      AbstractValue* output) {
                  this->CopyMpmPositions(context, output);
                },
                {systems::System<double>::xd_ticket()})
            .get_index();
  // all mpm related
  if (ExistsMpmModel()) {

    multibody::SpatialVelocity<double> velocity_sphere;
    velocity_sphere.translational() = Vector3<double>{0.1, 0.1, 0.1};
    velocity_sphere.rotational() = Vector3<double>{M_PI/2, M_PI/2, M_PI/2};

    mpm::SphereLevelSet level_set_sphere = mpm::SphereLevelSet(0.2);
    Vector3<double> translation_sphere = {0.0, 0.0, 0.0};
    math::RigidTransform<double> pose_sphere =
                            math::RigidTransform<double>(translation_sphere);

    double E = 5e4;
    double nu = 0.4;
    std::unique_ptr<mpm::CorotatedElasticModel> elastoplastic_model
            = std::make_unique<mpm::CorotatedElasticModel>(E, nu);
    typename mpm::MpmModel<T>::MaterialParameters m_param_sphere{
                                                std::move(elastoplastic_model),
                                                1000,
                                                velocity_sphere,
                                                1
                                                };

    mpm::Particles particles(0);

    InitializeParticles(level_set_sphere, pose_sphere, 
                        std::move(m_param_sphere), mpm_model_->grid_h(), particles);
    // InitializeParticles(level_set_sphere, pose_sphere, 
    //                     *(mpm_model_->material_params()), mpm_model_->grid_h(), particles);

    mpm::MpmState<T> mpm_state(particles);
    mpm_model_->num_particles_ = particles.get_num_particles();
    mpm_model_->particles_container_index_ = this->DeclareAbstractState(plant, Value<mpm::MpmState<T>>(mpm_state));
    particles_container_index_ = mpm_model_->particles_container_index_;
    // ---------------------------- add mpm_state to plant's context --------------

    
    // all mpm related
  } else {
    std::cout << "does not add mpm " << std::endl; getchar();
  }
  
  std::sort(body_ids_.begin(), body_ids_.end());
  for (DeformableBodyIndex i(0); i < static_cast<int>(body_ids_.size()); ++i) {
    DeformableBodyId id = body_ids_[i];
    body_id_to_index_[id] = i;
  }


}

template <typename T>
void DeformableModel<T>::CopyVertexPositions(const systems::Context<T>& context,
                                             AbstractValue* output) const {
  auto& output_value =
      output->get_mutable_value<geometry::GeometryConfigurationVector<T>>();
  output_value.clear();
  for (const auto& [body_id, geometry_id] : body_id_to_geometry_id_) {
    const auto& fem_model = GetFemModel(body_id);
    const int num_dofs = fem_model.num_dofs();
    const auto& discrete_state_index = GetDiscreteStateIndex(body_id);
    VectorX<T> vertex_positions =
        context.get_discrete_state(discrete_state_index).value().head(num_dofs);
    output_value.set_value(geometry_id, std::move(vertex_positions));
  }
}

template <typename T>
void DeformableModel<T>::CopyMpmPositions(const systems::Context<T>& context,
                                             AbstractValue* output) const {
  auto& output_value = output->get_mutable_value<std::vector<Vector3<double>>>();
  if (ExistsMpmModel()) {
    const mpm::MpmState<T>& current_state = context.template get_abstract_state<mpm::MpmState<T>>(particles_container_index_);
    const std::vector<Vector3<double>>& particle_positions = current_state.GetParticles().get_positions();
    output_value = particle_positions;
  } else {
    const std::vector<Vector3<double>>& particle_positions{}; // empty, port will still be connected anyways
    output_value = particle_positions;
  }
  
  
}

  // MODIFY particles in place!!
    // Initialize particles' positions with Poisson disk sampling. 
  template <typename T>
  void DeformableModel<T>::InitializeParticles(const mpm::AnalyticLevelSet& level_set,
                            const math::RigidTransform<double>& pose,
                            const typename mpm::MpmModel<T>::MaterialParameters& m_param, double grid_h, 
                            mpm::Particles& particles){

    DRAKE_DEMAND(m_param.density > 0.0);
    DRAKE_DEMAND(m_param.min_num_particles_per_cell >= 1);                        

    const std::array<Vector3<double>, 2> bounding_box = level_set.get_bounding_box();
    double sample_r =
            grid_h/(std::cbrt(m_param.min_num_particles_per_cell)+1);
    multibody::SpatialVelocity<double> init_v = m_param.initial_velocity;
    std::array<double, 3> xmin = {bounding_box[0][0], bounding_box[0][1],
                                  bounding_box[0][2]};
    std::array<double, 3> xmax = {bounding_box[1][0], bounding_box[1][1],
                                  bounding_box[1][2]};
    
    std::vector<Vector3<double>> particles_sample_positions =
        thinks::PoissonDiskSampling<double, 3, Vector3<double>>(sample_r,
                                                                xmin, xmax);

    // Pick out sampled particles that are in the object
    int num_samples = particles_sample_positions.size();
    std::vector<Vector3<double>> particles_positions, particles_velocities;
    for (int p = 0; p < num_samples; ++p) {
        const math::RigidTransform<double>& X_WB       = pose;
        const multibody::SpatialVelocity<double>& V_WB = init_v;
        const math::RotationMatrix<double>& Rot_WB     = X_WB.rotation();
        const Vector3<double>& p_BoBp_B = particles_sample_positions[p];
        const Vector3<double>& p_BoBp_W = Rot_WB*p_BoBp_B;
        multibody::SpatialVelocity<double> V_WBp   = V_WB.Shift(p_BoBp_W);

        // If the particle is in the level set
        if (level_set.InInterior(p_BoBp_B)) {
            // TODO(yiminlin.tri): Initialize the affine matrix C_p using
            //                     V_WBp.rotational() ?
            particles_velocities.emplace_back(V_WBp.translational());
            particles_positions.emplace_back(X_WB*p_BoBp_B);
        }
    }

    int num_particles = particles_positions.size();
    double reference_volume_p = level_set.get_volume()/num_particles;
    double init_m = m_param.density*reference_volume_p;

    // Add particles
    for (int p = 0; p < num_particles; ++p) {
        const Vector3<double>& xp = particles_positions[p];
        const Vector3<double>& vp = particles_velocities[p];
        Matrix3<double> elastic_deformation_grad_p = Matrix3<double>::Identity();
        Matrix3<double> kirchhoff_stress_p = Matrix3<double>::Identity();
        Matrix3<double> B_p                = Matrix3<double>::Zero();
        std::unique_ptr<mpm::ElastoPlasticModel> elastoplastic_model_p
                                        = m_param.elastoplastic_model->Clone();
        particles.AddParticle(xp, vp, init_m, reference_volume_p,
                               elastic_deformation_grad_p,
                               kirchhoff_stress_p,
                               B_p, std::move(elastoplastic_model_p));
    }                         
  }


template <typename T>
void DeformableModel<T>::ThrowUnlessRegistered(const char* source_method,
                                               DeformableBodyId id) const {
  if (fem_models_.find(id) == fem_models_.end()) {
    throw std::logic_error(std::string(source_method) +
                           "(): No deformable body with id " + to_string(id) +
                           " has been registered.");
  }
}

}  // namespace multibody
}  // namespace drake

template class drake::multibody::DeformableModel<double>;
