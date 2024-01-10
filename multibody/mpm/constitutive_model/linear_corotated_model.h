#pragma once
#include <memory>

#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

// TODO(zeshunzong): write doc for this class
template <typename T>
class LinearCorotatedModel : public ElastoPlasticModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(LinearCorotatedModel)

  LinearCorotatedModel(const T& youngs_modulus, const T& poissons_ratio);

  std::unique_ptr<ElastoPlasticModel<T>> Clone() const final {
    return std::make_unique<LinearCorotatedModel<T>>(*this);
  }

  void CalcFEFromFtrial(const Matrix3<T>& F_trial, Matrix3<T>* FE) const final;

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
