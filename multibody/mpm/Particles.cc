#include "drake/multibody/mpm/Particles.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
Particles<T>::Particles(): num_particles_(0) {}

template <typename T>
Particles<T>::Particles(int num_particles): num_particles_(num_particles),
                                         positions_(num_particles),
                                         velocities_(num_particles),
                                         masses_(num_particles),
                                         reference_volumes_(num_particles),
                                         elastic_deformation_gradients_(
                                                            num_particles),
                                         kirchhoff_stresses_(num_particles),
                                         first_PK_stresses_(num_particles),
                                         B_matrices_(num_particles),
                                         elastoplastic_models_(num_particles),
                                         bases_val_particles_(num_particles),
                                         bases_grad_particles_(num_particles) {
    DRAKE_ASSERT(num_particles >= 0);
}

template <typename T>
int Particles<T>::get_num_particles() const {
    return num_particles_;
}

template <typename T>
const Vector3<T>& Particles<T>::get_position(int index) const {
    return positions_[index];
}

template <typename T>
const Vector3<T>& Particles<T>::get_velocity(int index) const {
    return velocities_[index];
}

template <typename T>
const T& Particles<T>::get_mass(int index) const {
    return masses_[index];
}

template <typename T>
const T& Particles<T>::get_reference_volume(int index) const {
    return reference_volumes_[index];
}

template <typename T>
const Matrix3<T>& Particles<T>::get_elastic_deformation_gradient(int index)
                                                                        const {
    return elastic_deformation_gradients_[index];
}

template <typename T>
const Matrix3<T>& Particles<T>::get_elastic_deformation_gradient_new(int index)
                                                                        const {
    return elastic_deformation_gradients_new_[index];
}

template <typename T>
const Matrix3<T>& Particles<T>::get_kirchhoff_stress(int index) const {
    return kirchhoff_stresses_[index];
}

template <typename T>
const Matrix3<T>& Particles<T>::get_first_PK_stress(int index) const {
    return first_PK_stresses_[index];
}

template <typename T>
const Matrix3<T>& Particles<T>::get_B_matrix(int index) const {
    return B_matrices_[index];
}

template <typename T>
const std::vector<Vector3<T>>& Particles<T>::get_positions() const {
    return positions_;
}

template <typename T>
const std::vector<Vector3<T>>& Particles<T>::get_velocities() const {
    return velocities_;
}

template <typename T>
const std::vector<T>& Particles<T>::get_masses() const {
    return masses_;
}

template <typename T>
const std::vector<T>& Particles<T>::get_reference_volumes() const {
    return reference_volumes_;
}

template <typename T>
const std::vector<Matrix3<T>>&
                            Particles<T>::get_elastic_deformation_gradients()
                                                                        const {
    return elastic_deformation_gradients_;
}

template <typename T>
const std::vector<Matrix3<T>>& Particles<T>::get_kirchhoff_stresses() const {
    return kirchhoff_stresses_;
}

template <typename T>
const std::vector<Matrix3<T>>& Particles<T>::get_first_PK_stresses() const {
    return first_PK_stresses_;
}

template <typename T>
const std::vector<Matrix3<T>>& Particles<T>::get_B_matrices() const {
    return B_matrices_;
}

template <typename T>
void Particles<T>::set_position(int index, const Vector3<T>& position) {
    positions_[index] = position;
}

template <typename T>
void Particles<T>::set_velocity(int index, const Vector3<T>& velocity) {
    velocities_[index] = velocity;
}

template <typename T>
void Particles<T>::set_mass(int index, T mass) {
    DRAKE_DEMAND(mass > 0.0);
    masses_[index] = mass;
}

template <typename T>
void Particles<T>::set_reference_volume(int index, T reference_volume) {
    DRAKE_DEMAND(reference_volume > 0.0);
    reference_volumes_[index] = reference_volume;
}

template <typename T>
void Particles<T>::set_elastic_deformation_gradient(int index,
                        const Matrix3<T>& elastic_deformation_gradient) {
    elastic_deformation_gradients_[index] = elastic_deformation_gradient;
}

template <typename T>
void Particles<T>::set_elastic_deformation_gradient_new(int index,
                        const Matrix3<T>& elastic_deformation_gradient_new) {
    elastic_deformation_gradients_new_[index] = elastic_deformation_gradient_new;
}

template <typename T>
void Particles<T>::set_kirchhoff_stress(int index,
                                     const Matrix3<T>& kirchhoff_stress) {
    kirchhoff_stresses_[index] = kirchhoff_stress;
}

template <typename T>
void Particles<T>::set_first_PK_stress(int index,
                                     const Matrix3<T>& first_PK_stress) {
    first_PK_stresses_[index] = first_PK_stress;
}

template <typename T>
void Particles<T>::set_B_matrix(int index, const Matrix3<T>& B_matrix) {
    B_matrices_[index] = B_matrix;
}

template <typename T>
void Particles<T>::set_elastoplastic_model(int index,
                    std::unique_ptr<ElastoPlasticModel<T>> elastoplastic_model) {
    elastoplastic_models_[index] = std::move(elastoplastic_model);
}

template <typename T>
void Particles<T>::set_positions(const std::vector<Vector3<T>>& positions) {
    positions_ = positions;
}

template <typename T>
void Particles<T>::set_velocities(const std::vector<Vector3<T>>& velocities) {
    velocities_ = velocities;
}

template <typename T>
void Particles<T>::set_masses(const std::vector<T>& masses) {
    masses_ = masses;
}

template <typename T>
void Particles<T>::set_reference_volumes(const std::vector<T>&
                                        reference_volumes) {
    reference_volumes_ = reference_volumes;
}

template <typename T>
void Particles<T>::set_elastic_deformation_gradients(
            const std::vector<Matrix3<T>>& elastic_deformation_gradients) {
    elastic_deformation_gradients_ = elastic_deformation_gradients;
}

template <typename T>
void Particles<T>::set_kirchhoff_stresses(const std::vector<Matrix3<T>>&
                            kirchhoff_stresses) {
    kirchhoff_stresses_ = kirchhoff_stresses;
}

template <typename T>
void Particles<T>::set_first_PK_stresses(const std::vector<Matrix3<T>>&
                            first_PK_stresses) {
    first_PK_stresses_ = first_PK_stresses;
}

template <typename T>
void Particles<T>::set_B_matrices(const std::vector<Matrix3<T>>& B_matrices) {
    B_matrices_ = B_matrices;
}

template <typename T>
void Particles<T>::Reorder(const std::vector<size_t>& new_order) {
    DRAKE_DEMAND(static_cast<int>(new_order.size()) == num_particles_);
    int p_new;
    std::vector<Vector3<T>> positions_sorted(num_particles_);
    std::vector<Vector3<T>> velocities_sorted(num_particles_);
    std::vector<T> masses_sorted(num_particles_);
    std::vector<T> reference_volumes_sorted(num_particles_);
    std::vector<Matrix3<T>>
                        elastic_deformation_gradients_sorted(num_particles_);
    std::vector<Matrix3<T>> kirchhoff_stresses_sorted(num_particles_);
    std::vector<Matrix3<T>> first_PK_stresses_sorted(num_particles_);
    std::vector<Matrix3<T>> B_matrices_sorted(num_particles_);
    std::vector<copyable_unique_ptr<ElastoPlasticModel<T>>>
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
        first_PK_stresses_sorted[p]             = first_PK_stresses_[p_new];
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
    first_PK_stresses_.swap(first_PK_stresses_sorted);
    B_matrices_.swap(B_matrices_sorted);
    elastoplastic_models_.swap(elastoplastic_models_sorted);
}

template <typename T>
void Particles<T>::AddParticle(const Vector3<T>& position,
                            const Vector3<T>& velocity,
                            T mass, T reference_volume,
                            const Matrix3<T>& elastic_deformation_gradient,
                            const Matrix3<T>& kirchhoff_stress,
                            const Matrix3<T>& first_PK_stress,
                            const Matrix3<T>& B_matrix,
                    std::unique_ptr<ElastoPlasticModel<T>> elastoplastic_model) {
    positions_.emplace_back(position);
    velocities_.emplace_back(velocity);
    masses_.emplace_back(mass);
    reference_volumes_.emplace_back(reference_volume);
    elastic_deformation_gradients_.emplace_back(elastic_deformation_gradient);
    kirchhoff_stresses_.emplace_back(kirchhoff_stress);
    first_PK_stresses_.emplace_back(first_PK_stress);
    B_matrices_.emplace_back(B_matrix);
    elastoplastic_models_.emplace_back(std::move(elastoplastic_model));
    bases_val_particles_.emplace_back(std::array<T, 27>{});
    bases_grad_particles_.emplace_back(std::array<Vector3<T>, 27>{});
    num_particles_++;
}

template <typename T>
void Particles<T>::ApplyPlasticityAndUpdateStresses() {
    for (int p = 0; p < num_particles_; ++p) {
        elastoplastic_models_[p]->
                            UpdateDeformationGradientAndCalcKirchhoffStress(
                                            &kirchhoff_stresses_[p],
                                            &elastic_deformation_gradients_[p]);
        elastoplastic_models_[p]->
                            CalcFirstPiolaStress(
                                            elastic_deformation_gradients_[p],
                                            &first_PK_stresses_[p]);
    }
}


template <typename T>
void Particles<T>::AdvectParticles(T dt) {
    for (int p = 0; p < num_particles_; ++p) {
        positions_[p] += dt*velocities_[p];
    }
}

template <typename T>
void Particles<T>::ComputePiolaDerivatives() {
    DRAKE_DEMAND(static_cast<int>(elastic_deformation_gradients_new_.size()) == num_particles_);    
    for (int p = 0; p < num_particles_; ++p) {
        elastoplastic_models_[p]->CalcFirstPiolaStressDerivative(elastic_deformation_gradients_new_[p], &stress_derivatives_[p]);
    }
}

/* 
For each particle p, compute and store Result(3β+α, 3γ+ρ) = [∑ᵢⱼ (dPₐᵢ/dFᵨⱼ) * Fᵧⱼ * Fᵦᵢ] * Vₚ⁰
i and j are θ and ϕ in accompanied ElasticEnergyDerivatives.md
*/
template <typename T>
void Particles<T>::ContractPiolaDerivativesWithFWithF() {
    DRAKE_DEMAND(static_cast<int>(stress_derivatives_contractF_contractF_.size()) == num_particles_);
    for (int index = 0; index < num_particles_; ++index) {
        stress_derivatives_contractF_contractF_[index].setZero();
        Eigen::Matrix3<T> Fp0 = elastic_deformation_gradients_[index];
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        for (int gamma = 0; gamma < 3; ++gamma) {
                            for (int rho = 0; rho < 3; ++rho) {
                                stress_derivatives_contractF_contractF_[index](3*beta+alpha, 3*gamma+rho) +=  stress_derivatives_[index](3*i+alpha, 3*j+rho) * Fp0(gamma,j) * Fp0(beta,i) * get_reference_volume(index);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
TotalMassEnergyMomentum<T> Particles<T>::GetTotalMassEnergyMomentum(T g) const {
    TotalMassEnergyMomentum<T> sum_particles_state;
    // Particles' sum of mass and momentum
    sum_particles_state.sum_mass             = 0.0;
    sum_particles_state.sum_kinetic_energy   = 0.0;
    sum_particles_state.sum_strain_energy    = 0.0;
    sum_particles_state.sum_potential_energy = 0.0;
    sum_particles_state.sum_momentum         = {0.0, 0.0, 0.0};
    sum_particles_state.sum_angular_momentum = {0.0, 0.0, 0.0};
    for (int p = 0; p < num_particles_; ++p) {
        T mp   = masses_[p];
        T volp = reference_volumes_[p];
        const Vector3<T>& vp = velocities_[p];
        const Vector3<T>& xp = positions_[p];
        const Matrix3<T>& Bp = B_matrices_[p];
        const Matrix3<T>& Fp = elastic_deformation_gradients_[p]; 
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
                        + mathutils::ContractionWithLeviCivita<T>(Bp.transpose()));
    }
    return sum_particles_state;
}


template class Particles<double>;
template class Particles<AutoDiffXd>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
