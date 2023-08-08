#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <iostream>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"
#include "drake/multibody/mpm/MathUtils.h"
#include "drake/multibody/mpm/TotalMassEnergyMomentum.h"

namespace drake {
namespace multibody {
namespace mpm {


// A particles class holding vectors of particles' state
template <typename T>
class Particles {
 public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Particles);
    Particles();
    explicit Particles(int num_particles);


    int get_num_particles() const;
    // Note that we didn't overload get_position: get_positions for getting
    // the whole vector
    const Vector3<T>& get_position(int index) const;
    const Vector3<T>& get_velocity(int index) const;
    const T& get_mass(int index) const;
    const T& get_reference_volume(int index) const;
    const Matrix3<T>& get_elastic_deformation_gradient(int index) const;
    const Matrix3<T>& get_elastic_deformation_gradient_new(int index) const;
    const Matrix3<T>& get_kirchhoff_stress(int index) const;
    const Matrix3<T>& get_first_PK_stress(int index) const;
    const Matrix3<T>& get_B_matrix(int index) const;

    const std::vector<Vector3<T>>& get_positions() const;
    const std::vector<Vector3<T>>& get_velocities() const;
    const std::vector<T>& get_masses() const;
    const std::vector<T>& get_reference_volumes() const;
    const std::vector<Matrix3<T>>& get_elastic_deformation_gradients()const;

    const std::vector<Matrix3<T>>& get_kirchhoff_stresses() const;
    const std::vector<Matrix3<T>>& get_first_PK_stresses() const;
    // Get the matrix B_p, who composes the affine matrix C_p in APIC:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    const std::vector<Matrix3<T>>& get_B_matrices() const;

    void print_info() const {
      for (int ind = 0; ind < num_particles_; ++ind){
         if (ind < 10){
            std::cout << "particle " << ind << " position " << positions_[ind][0] << " " << positions_[ind][1] << " " << positions_[ind][2]<< std::endl;
         }
      }
    }

    // TODO(yiminlin.tri): To this point, the encapsulation seems useless here,
    // Maybe directly make Particles as a struct and remove setters and getters?
    // TODO(yiminlin.tri): ideally thfem
    void set_position(int index, const Vector3<T>& position);
    void set_velocity(int index, const Vector3<T>& velocity);
    void set_mass(int index, T mass);
    void set_reference_volume(int index, T reference_volume);
    void set_elastic_deformation_gradient(int index,
                           const Matrix3<T>& elastic_deformation_gradient);
   
    void resize_elastic_deformation_gradient_new(size_t s) {
      elastic_deformation_gradients_new_.resize(s);
    }

    void set_elastic_deformation_gradient_new(int index,
                        const Matrix3<T>& elastic_deformation_gradient_new);

    void resize_stress_derivatives(size_t s) {
      stress_derivatives_.resize(s);
    }

    void resize_stress_derivatives_contractF_contractF(size_t s) {
      stress_derivatives_contractF_contractF_.resize(s);
    }

    T CalcParticleEnergyDensity(int index, const Matrix3<T>& Fp) const {
      return elastoplastic_models_[index]->CalcStrainEnergyDensity(Fp);
    }

    void compute_first_PK_stress(int index, const Matrix3<T>& FE, Matrix3<T>* PK_stress) const {
      elastoplastic_models_[index]->CalcFirstPiolaStress(FE, PK_stress);
    }         

    void set_kirchhoff_stress(int index,
                              const Matrix3<T>& kirchhoff_stress);
    void set_first_PK_stress(int index,
                              const Matrix3<T>& first_PK_stress);
                              
   

    void set_B_matrix(int index, const Matrix3<T>& B_matrix);
    void set_elastoplastic_model(int index,
                     std::unique_ptr<ElastoPlasticModel<T>> elastoplastic_model);

    void set_positions(const std::vector<Vector3<T>>& positions);
    void set_velocities(const std::vector<Vector3<T>>& velocities);
    void set_masses(const std::vector<T>& masses);
    void set_reference_volumes(const std::vector<T>& reference_volumes);
    void set_elastic_deformation_gradients(const std::vector<Matrix3<T>>&
                                           elastic_deformation_gradients);
    void set_kirchhoff_stresses(const std::vector<Matrix3<T>>&
                                kirchhoff_stresses);
    void set_first_PK_stresses(const std::vector<Matrix3<T>>&
                                first_PK_stresses);
    // Set the matrix B_p, who composes the affine matrix C_p in APIC:
    // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p),
    void set_B_matrices(const std::vector<Matrix3<T>>& B_matrices);

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
    void AddParticle(const Vector3<T>& position,
                     const Vector3<T>& velocity,
                     T mass, T reference_volume,
                     const Matrix3<T>& elastic_deformation_gradient,
                     const Matrix3<T>& kirchhoff_stress,
                     const Matrix3<T>& first_PK_stress,
                     const Matrix3<T>& B_matrix,
                     std::unique_ptr<ElastoPlasticModel<T>> elastoplastic_model);

    // Assume the elastic deformation gradients are in their trial state
    // (Fₑ = Fₑᵗʳⁱᵃˡ), update the elastic deformation gradients by projecting
    // the elastic deformation gradient to the yield surface, and update
    // Kirchhoff stress AS WELL AS FIRST_PK_stress with the constitutive relation.
    void ApplyPlasticityAndUpdateStresses();

    // Particle advection using the updated velocities, assuming they are
    // already updated in the member variables.
    void AdvectParticles(T dt);

    // For each particle p, after elastic_deformation_gradients_new_[p] is computed, compute dPdF and store it.
    void ComputePiolaDerivatives();


    // For each particle p, after dPdF is computed, compute dPdF contract F⁰ₚ contract F⁰ₚ * Vₚ⁰ and store it.
    void ContractPiolaDerivativesWithFWithF();

    Eigen::Matrix<T, 9, 9>& get_stress_derivatives_contractF_contractF_(int index) {
      return stress_derivatives_contractF_contractF_[index];
    }

    // Return the sum of mass, momentum and angular momentum of all particles.
    // The sum of particles' angular momentums is ∑ mp xp×vp + Bp^T:ϵ
    // by https://math.ucdavis.edu/~jteran/papers/JST17.pdf section 5.3.1
    TotalMassEnergyMomentum<T> GetTotalMassEnergyMomentum(T g) const;


    // Note that the following weights and weight gradients will change if particle position changes 
    void SetWeightAtParticle(int index_particle, int index_neighbor_node, const T& weight) {
      DRAKE_DEMAND(index_neighbor_node>=0);
      DRAKE_DEMAND(index_neighbor_node<27); // each particle has 27 neighbor grid nodes
      bases_val_particles_[index_particle][index_neighbor_node] = weight;
    }
    T GetWeightAtParticle(int index_particle, int index_neighbor_node) const {
      DRAKE_DEMAND(index_neighbor_node>=0);
      DRAKE_DEMAND(index_neighbor_node<27); // each particle has 27 neighbor grid nodes
      return bases_val_particles_[index_particle][index_neighbor_node];
    }

    void SetNeighborGridGlobalIndex(int index_particle, int index_neighbor_node_local, size_t index_neighbor_node_global) {
      DRAKE_DEMAND(index_neighbor_node_local>=0);
      DRAKE_DEMAND(index_neighbor_node_local<27); // each particle has 27 neighbor grid nodes
      neighbor_grid_nodes_global_indices_[index_particle][index_neighbor_node_local] = index_neighbor_node_global;
    }

    void SetWeightGradientAtParticle(int index_particle, int index_neighbor_node, const Vector3<T>& dweight) {
      DRAKE_DEMAND(index_neighbor_node>=0);
      DRAKE_DEMAND(index_neighbor_node<27); // each particle has 27 neighbor grid nodes
      bases_grad_particles_[index_particle][index_neighbor_node] = dweight;
    }
    void GetWeightGradientAtParticle(int index_particle, int index_neighbor_node, Vector3<T>* dweight) {
      DRAKE_DEMAND(index_neighbor_node>=0);
      DRAKE_DEMAND(index_neighbor_node<27); // each particle has 27 neighbor grid nodes
      *dweight = bases_grad_particles_[index_particle][index_neighbor_node];
    }

 private:
    int num_particles_;
    std::vector<Vector3<T>> positions_{};
    std::vector<Vector3<T>> velocities_{};
    std::vector<T> masses_{};
    std::vector<T> reference_volumes_{};
    std::vector<Matrix3<T>> elastic_deformation_gradients_{};
    std::vector<Matrix3<T>> elastic_deformation_gradients_new_{}; // as a function of grid v
    std::vector<Matrix3<T>> kirchhoff_stresses_{};
    std::vector<Matrix3<T>> first_PK_stresses_{};
    // The affine matrix B_p in APIC
    std::vector<Matrix3<T>> B_matrices_{};
    std::vector<copyable_unique_ptr<ElastoPlasticModel<T>>> elastoplastic_models_{};

    std::vector<Eigen::Matrix<T, 9, 9>> stress_derivatives_{};
    std::vector<Eigen::Matrix<T, 9, 9>> stress_derivatives_contractF_contractF_{};

    // each particle p is related to its 27 neighbor grid nodes. This stores the corresponding weights
    // wip. This is the same as the one in MPMTransfer. Should consider deprecate one of them in the future.
    std::vector<std::array<T, 27>> bases_val_particles_{};
    std::vector<std::array<Vector3<T>, 27>> bases_grad_particles_{};
    std::vector<std::array<size_t, 27>> neighbor_grid_nodes_global_indices_{};



    
};  // class Particles

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
