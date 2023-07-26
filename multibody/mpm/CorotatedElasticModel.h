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
template <typename T>
class CorotatedElasticModel : public ElastoPlasticModel<T> {
 public:
    CorotatedElasticModel();
    CorotatedElasticModel(T E, T nu);

    virtual std::unique_ptr<ElastoPlasticModel<T>> Clone() const {
        return std::make_unique<CorotatedElasticModel<T>>(*this);
    }

    T CalcStrainEnergyDensity(const Matrix3<T>& FE) const final;

    void CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const;

    void CalcFirstPiolaStressDerivative(const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const;

    void UpdateDeformationGradientAndCalcKirchhoffStress(
                    Matrix3<T>* tau,
                    Matrix3<T>* elastic_deformation_gradient) const final;
};  // class ElastoPlasticModel

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

