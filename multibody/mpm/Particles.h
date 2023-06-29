#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/MathUtils.h"
#include "drake/multibody/mpm/TotalMassEnergyMomentum.h"

namespace drake {
namespace multibody {
namespace mpm {

// A particles class holding vectors of particles' state
class Particles {
 public:
    Particles();
    explicit Particles(int num_particles);

    int get_num_particles() const;
    // Note that we didn't overload get_position: get_positions for getting
    // the whole vector
    const Vector3<double>& get_position(int index) const;
    const Vector3<double>& get_velocity(int index) const;
    const double& get_mass(int index) const;
    const double& get_reference_volume(int index) const;
    const Matrix3<double>& get_elastic_deformation_gradient(int index) const;
    const Matrix3<double>& get_kirchhoff_stress(int index) const;
    const Matrix3<double>& get_B_matrix(int index) const;

    const std::vector<Vector3<double>>& get_positions() const;
    const std::vector<Vector3<double>>& get_velocities() const;
    const std::vector<double>& get_masses() const;
    const std::vector<double>& get_reference_volumes() const;
    const std::vector<Matrix3<double>>& get_elastic_deformation_gradients()
                                                                        const;
    const std::vector<Matrix3<double>>& get_kirchhoff_stresses() const;
    // Get the matrix B_p, who composes the affine matrix C_p in APIC:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    const std::vector<Matrix3<double>>& get_B_matrices() const;

    // TODO(yiminlin.tri): To this point, the encapsulation seems useless here,
    // Maybe directly make Particles as a struct and remove setters and getters?
    // TODO(yiminlin.tri): ideally thfem
    void set_position(int index, const Vector3<double>& position);
    void set_velocity(int index, const Vector3<double>& velocity);
    void set_mass(int index, double mass);
    void set_reference_volume(int index, double reference_volume);
    void set_elastic_deformation_gradient(int index,
                           const Matrix3<double>& elastic_deformation_gradient);
    void set_kirchhoff_stress(int index,
                              const Matrix3<double>& kirchhoff_stress);
    void set_B_matrix(int index, const Matrix3<double>& B_matrix);
    void set_elastoplastic_model(int index,
                     std::unique_ptr<ElastoPlasticModel> elastoplastic_model);

    void set_positions(const std::vector<Vector3<double>>& positions);
    void set_velocities(const std::vector<Vector3<double>>& velocities);
    void set_masses(const std::vector<double>& masses);
    void set_reference_volumes(const std::vector<double>& reference_volumes);
    void set_elastic_deformation_gradients(const std::vector<Matrix3<double>>&
                                           elastic_deformation_gradients);
    void set_kirchhoff_stresses(const std::vector<Matrix3<double>>&
                                kirchhoff_stresses);
    // Set the matrix B_p, who composes the affine matrix C_p in APIC:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    void set_B_matrices(const std::vector<Matrix3<double>>& B_matrices);

    // TODO(yiminlin.tri): in place sorting
    // Permute all states in the Particles with respect to the index set
    // new_order. e.g. given new_order = [2; 0; 1], and the original
    // particle states are denoted by [p0; p1; p2]. The new particles states
    // after calling Reorder will be [p2; p0; p1]
    // @pre new_order is a permutation of [0, ..., new_order.size()-1]
    void Reorder(const std::vector<size_t>& new_order);

    // Add a particle with the given properties. The default material
    // is dough with Young's modulus E = 9e4 and Poisson ratio nu = 0.49.
    // B_matrix denotes matrix B_p, who composes the affine matrix C_p in APIC:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    void AddParticle(const Vector3<double>& position,
                     const Vector3<double>& velocity,
                     double mass, double reference_volume,
                     const Matrix3<double>& elastic_deformation_gradient,
                     const Matrix3<double>& kirchhoff_stress,
                     const Matrix3<double>& B_matrix,
                     std::unique_ptr<ElastoPlasticModel> elastoplastic_model);

    // Assume the elastic deformation gradients are in their trial state
    // (Fₑ = Fₑᵗʳⁱᵃˡ), update the elastic deformation gradients by projecting
    // the elastic deformation gradient to the yield surface, and update
    // Kirchhoff stress with the constitutive relation.
    void ApplyPlasticityAndUpdateKirchhoffStresses();

    // Particle advection using the updated velocities, assuming they are
    // already updated in the member variables.
    void AdvectParticles(double dt);

    // Return the sum of mass, momentum and angular momentum of all particles.
    // The sum of particles' angular momentums is ∑ mp xp×vp + Bp^T:ϵ
    // by https://math.ucdavis.edu/~jteran/papers/JST17.pdf section 5.3.1
    TotalMassEnergyMomentum GetTotalMassEnergyMomentum(double g) const;

 private:
    int num_particles_;
    std::vector<Vector3<double>> positions_{};
    std::vector<Vector3<double>> velocities_{};
    std::vector<double> masses_{};
    std::vector<double> reference_volumes_{};
    std::vector<Matrix3<double>> elastic_deformation_gradients_{};
    std::vector<Matrix3<double>> kirchhoff_stresses_{};
    // The affine matrix B_p in APIC
    std::vector<Matrix3<double>> B_matrices_{};
    std::vector<std::unique_ptr<ElastoPlasticModel>> elastoplastic_models_{};
};  // class Particles

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
