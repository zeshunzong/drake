#pragma once
#include <memory>

#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

// Implements the fixed corotated hyperelastic constitutive model as described
// in [Stomakhin, 2012]. [Stomakhin, 2012] Stomakhin, Alexey, et al.
// "Energetically consistent invertible elasticity." Proceedings of the 11th ACM
// SIGGRAPH/Eurographics conference on Computer Animation. 2012.
template <typename T>
class CorotatedElasticModel : public ElastoPlasticModel<T> {
 public:
  CorotatedElasticModel(const T& youngs_modulus, const T& poissons_ratio);

  std::unique_ptr<ElastoPlasticModel<T>> Clone() const final {
    return std::make_unique<CorotatedElasticModel<T>>(*this);
  }

  void CalcFEFromFtrial(Matrix3<T>* F_trial) const final;

  T CalcStrainEnergyDensity(const Matrix3<T>& FE) const final;

  void CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const final;

  void CalcKirchhoffStress(const Matrix3<T>& FE, Matrix3<T>* tau) const final;

  void CalcFirstPiolaStressDerivative(const Matrix3<T>& FE,
                                      Eigen::Matrix<T, 9, 9>* dPdF) const final;
};

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
