#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace mpm {

// A base class providing the interface of constituive and plastic model
template <typename T>
class ElastoPlasticModel {
 public:
    // Default material, dough: E = 9e4 Pa, nu = 0.49
    ElastoPlasticModel();

    // Constructor uses Young's modulus E and Poisson's ratio nu
    ElastoPlasticModel(T E, T nu);

    virtual std::unique_ptr<ElastoPlasticModel<T>> Clone() const = 0;

    T get_lambda() const {  return lambda_;  };
    T get_mu() const     {  return mu_;  };

    virtual T CalcStrainEnergyDensity(const Matrix3<T>& FE) const = 0;

    // Update the elastic deformation gradient according to the plasticity model
    // by projecting the trial elastic stress to the yield surface. Then
    // calculate the projected Kirchhoff stress by the projected deformation
    // gradient. Kirchhoff stress density is defined as tau = P Fᴱ^T
    //                                                      = dψ/dFᴱ Fᴱ^T,
    // where ψ denotes the energy density
    virtual void UpdateDeformationGradientAndCalcKirchhoffStress(
                    Matrix3<T>* tau,
                    Matrix3<T>* elastic_deformation_gradient) const = 0;

    virtual ~ElastoPlasticModel() = default;

 protected:
    T lambda_;                       // The first Lamé coefficient
    T mu_;                           // The second Lamé coefficient
};  // class ElastoPlasticModel

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
