#include "drake/multibody/mpm/Particles.h"

namespace drake {
namespace multibody {
namespace mpm {

Particles::Particles(): num_particles_(0) {}

Particles::Particles(int num_particles): num_particles_(num_particles),
                                         positions_(num_particles),
                                         velocities_(num_particles),
                                         masses_(num_particles),
                                         reference_volumes_(num_particles),
                                         elastic_deformation_gradients_(
                                                            num_particles),
                                         kirchhoff_stresses_(num_particles),
                                         B_matrices_(num_particles),
                                         elastoplastic_models_(num_particles) {
    DRAKE_ASSERT(num_particles >= 0);
}

int Particles::get_num_particles() const {
    return num_particles_;
}

const Vector3<double>& Particles::get_position(int index) const {
    return positions_[index];
}

const Vector3<double>& Particles::get_velocity(int index) const {
    return velocities_[index];
}

const double& Particles::get_mass(int index) const {
    return masses_[index];
}

const double& Particles::get_reference_volume(int index) const {
    return reference_volumes_[index];
}

const Matrix3<double>& Particles::get_elastic_deformation_gradient(int index)
                                                                        const {
    return elastic_deformation_gradients_[index];
}

const Matrix3<double>& Particles::get_kirchhoff_stress(int index) const {
    return kirchhoff_stresses_[index];
}

const Matrix3<double>& Particles::get_B_matrix(int index) const {
    return B_matrices_[index];
}

const std::vector<Vector3<double>>& Particles::get_positions() const {
    return positions_;
}

const std::vector<Vector3<double>>& Particles::get_velocities() const {
    return velocities_;
}

const std::vector<double>& Particles::get_masses() const {
    return masses_;
}

const std::vector<double>& Particles::get_reference_volumes() const {
    return reference_volumes_;
}

const std::vector<Matrix3<double>>&
                            Particles::get_elastic_deformation_gradients()
                                                                        const {
    return elastic_deformation_gradients_;
}

const std::vector<Matrix3<double>>& Particles::get_kirchhoff_stresses() const {
    return kirchhoff_stresses_;
}

const std::vector<Matrix3<double>>& Particles::get_B_matrices() const {
    return B_matrices_;
}

void Particles::set_position(int index, const Vector3<double>& position) {
    positions_[index] = position;
}

void Particles::set_velocity(int index, const Vector3<double>& velocity) {
    velocities_[index] = velocity;
}

void Particles::set_mass(int index, double mass) {
    DRAKE_DEMAND(mass > 0.0);
    masses_[index] = mass;
}

void Particles::set_reference_volume(int index, double reference_volume) {
    DRAKE_DEMAND(reference_volume > 0.0);
    reference_volumes_[index] = reference_volume;
}

void Particles::set_elastic_deformation_gradient(int index,
                        const Matrix3<double>& elastic_deformation_gradient) {
    elastic_deformation_gradients_[index] = elastic_deformation_gradient;
}

void Particles::set_kirchhoff_stress(int index,
                                     const Matrix3<double>& kirchhoff_stress) {
    kirchhoff_stresses_[index] = kirchhoff_stress;
}

void Particles::set_B_matrix(int index, const Matrix3<double>& B_matrix) {
    B_matrices_[index] = B_matrix;
}

void Particles::set_elastoplastic_model(int index,
                    std::unique_ptr<ElastoPlasticModel> elastoplastic_model) {
    elastoplastic_models_[index] = std::move(elastoplastic_model);
}

void Particles::set_positions(const std::vector<Vector3<double>>& positions) {
    positions_ = positions;
}

void Particles::set_velocities(const std::vector<Vector3<double>>& velocities) {
    velocities_ = velocities;
}

void Particles::set_masses(const std::vector<double>& masses) {
    masses_ = masses;
}

void Particles::set_reference_volumes(const std::vector<double>&
                                        reference_volumes) {
    reference_volumes_ = reference_volumes;
}

void Particles::set_elastic_deformation_gradients(
            const std::vector<Matrix3<double>>& elastic_deformation_gradients) {
    elastic_deformation_gradients_ = elastic_deformation_gradients;
}

void Particles::set_kirchhoff_stresses(const std::vector<Matrix3<double>>&
                            kirchhoff_stresses) {
    kirchhoff_stresses_ = kirchhoff_stresses;
}

void Particles::set_B_matrices(const std::vector<Matrix3<double>>& B_matrices) {
    B_matrices_ = B_matrices;
}

void Particles::Reorder(const std::vector<size_t>& new_order) {
    DRAKE_DEMAND(static_cast<int>(new_order.size()) == num_particles_);
    int p_new;
    std::vector<Vector3<double>> positions_sorted(num_particles_);
    std::vector<Vector3<double>> velocities_sorted(num_particles_);
    std::vector<double> masses_sorted(num_particles_);
    std::vector<double> reference_volumes_sorted(num_particles_);
    std::vector<Matrix3<double>>
                        elastic_deformation_gradients_sorted(num_particles_);
    std::vector<Matrix3<double>> kirchhoff_stresses_sorted(num_particles_);
    std::vector<Matrix3<double>> B_matrices_sorted(num_particles_);
    std::vector<std::unique_ptr<ElastoPlasticModel>>
                                 elastoplastic_models_sorted(num_particles_);
    for (int p = 0; p < num_particles_; ++p) {
        p_new = new_order[p];
        positions_sorted[p]                     = positions_[p_new];
        velocities_sorted[p]                    = velocities_[p_new];
        masses_sorted[p]                        = masses_[p_new];
        reference_volumes_sorted[p]             = reference_volumes_[p_new];
        elastic_deformation_gradients_sorted[p] =
                                        elastic_deformation_gradients_[p_new];
        kirchhoff_stresses_sorted[p]            = kirchhoff_stresses_[p_new];
        B_matrices_sorted[p]                    = B_matrices_[p_new];
        elastoplastic_models_sorted[p]          =
                                        std::move(elastoplastic_models_[p_new]);
    }
    positions_.swap(positions_sorted);
    velocities_.swap(velocities_sorted);
    masses_.swap(masses_sorted);
    reference_volumes_.swap(reference_volumes_sorted);
    elastic_deformation_gradients_.swap(elastic_deformation_gradients_sorted);
    kirchhoff_stresses_.swap(kirchhoff_stresses_sorted);
    B_matrices_.swap(B_matrices_sorted);
    elastoplastic_models_.swap(elastoplastic_models_sorted);
}

void Particles::AddParticle(const Vector3<double>& position,
                            const Vector3<double>& velocity,
                            double mass, double reference_volume,
                            const Matrix3<double>& elastic_deformation_gradient,
                            const Matrix3<double>& kirchhoff_stress,
                            const Matrix3<double>& B_matrix,
                    std::unique_ptr<ElastoPlasticModel> elastoplastic_model) {
    positions_.emplace_back(position);
    velocities_.emplace_back(velocity);
    masses_.emplace_back(mass);
    reference_volumes_.emplace_back(reference_volume);
    elastic_deformation_gradients_.emplace_back(elastic_deformation_gradient);
    kirchhoff_stresses_.emplace_back(kirchhoff_stress);
    B_matrices_.emplace_back(B_matrix);
    elastoplastic_models_.emplace_back(std::move(elastoplastic_model));
    num_particles_++;
}

void Particles::ApplyPlasticityAndUpdateKirchhoffStresses() {
    for (int p = 0; p < num_particles_; ++p) {
        elastoplastic_models_[p]->
                            UpdateDeformationGradientAndCalcKirchhoffStress(
                                            &kirchhoff_stresses_[p],
                                            &elastic_deformation_gradients_[p]);
    }
}

void Particles::AdvectParticles(double dt) {
    for (int p = 0; p < num_particles_; ++p) {
        positions_[p] += dt*velocities_[p];
    }
}

TotalMassEnergyMomentum Particles::GetTotalMassEnergyMomentum(double g) const {
    TotalMassEnergyMomentum sum_particles_state;
    // Particles' sum of mass and momentum
    sum_particles_state.sum_mass             = 0.0;
    sum_particles_state.sum_kinetic_energy   = 0.0;
    sum_particles_state.sum_strain_energy    = 0.0;
    sum_particles_state.sum_potential_energy = 0.0;
    sum_particles_state.sum_momentum         = {0.0, 0.0, 0.0};
    sum_particles_state.sum_angular_momentum = {0.0, 0.0, 0.0};
    for (int p = 0; p < num_particles_; ++p) {
        double mp   = masses_[p];
        double volp = reference_volumes_[p];
        const Vector3<double>& vp = velocities_[p];
        const Vector3<double>& xp = positions_[p];
        const Matrix3<double>& Bp = B_matrices_[p];
        const Matrix3<double>& Fp = elastic_deformation_gradients_[p]; 
        sum_particles_state.sum_mass             += mp;
        sum_particles_state.sum_kinetic_energy   += .5*mp*vp.dot(vp);
        sum_particles_state.sum_strain_energy    += volp*
                                                    elastoplastic_models_[p]
                                                  ->CalcStrainEnergyDensity(Fp);
        // TODO(yiminlin.tri): hardcoded, assume g is of form (0, 0, g)
        //                     need to avoid pass by argument
        sum_particles_state.sum_potential_energy += -mp*g*xp[2];
        sum_particles_state.sum_momentum         += mp*vp;
        sum_particles_state.sum_angular_momentum += mp*(xp.cross(vp)
                        + mathutils::ContractionWithLeviCivita(Bp.transpose()));
    }
    return sum_particles_state;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
