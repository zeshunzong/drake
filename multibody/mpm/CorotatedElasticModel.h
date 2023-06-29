#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/mpm/ElastoPlasticModel.h"

namespace drake {
namespace multibody {
namespace mpm {

// A implementation of Fixed Corotated Model (Constitutive Model), without
// plasticity
class CorotatedElasticModel : public ElastoPlasticModel {
 public:
    CorotatedElasticModel();
    CorotatedElasticModel(double E, double nu);

    virtual std::unique_ptr<ElastoPlasticModel> Clone() const {
        return std::make_unique<CorotatedElasticModel>(*this);
    }

    double CalcStrainEnergyDensity(const Matrix3<double>& FE) const final;

    void UpdateDeformationGradientAndCalcKirchhoffStress(
                    Matrix3<double>* tau,
                    Matrix3<double>* elastic_deformation_gradient) const final;
};  // class ElastoPlasticModel

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

