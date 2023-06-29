#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace mpm {

// A base class providing the interface of constituive and plastic model
class ElastoPlasticModel {
 public:
    // Default material, dough: E = 9e4 Pa, nu = 0.49
    ElastoPlasticModel();

    // Constructor uses Young's modulus E and Poisson's ratio nu
    ElastoPlasticModel(double E, double nu);

    virtual std::unique_ptr<ElastoPlasticModel> Clone() const = 0;

    double get_lambda() const {  return lambda_;  };
    double get_mu() const     {  return mu_;  };

    virtual double CalcStrainEnergyDensity(const Matrix3<double>& FE) const = 0;

    // Update the elastic deformation gradient according to the plasticity model
    // by projecting the trial elastic stress to the yield surface. Then
    // calculate the projected Kirchhoff stress by the projected deformation
    // gradient. Kirchhoff stress density is defined as tau = P Fᴱ^T
    //                                                      = dψ/dFᴱ Fᴱ^T,
    // where ψ denotes the energy density
    virtual void UpdateDeformationGradientAndCalcKirchhoffStress(
                    Matrix3<double>* tau,
                    Matrix3<double>* elastic_deformation_gradient) const = 0;

    virtual ~ElastoPlasticModel() = default;

 protected:
    double lambda_;                       // The first Lamé coefficient
    double mu_;                           // The second Lamé coefficient
};  // class ElastoPlasticModel

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
