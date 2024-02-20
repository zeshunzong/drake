#pragma once
#include <memory>
#include <iostream>
#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

// TODO(zeshunzong): write doc for this class
template <typename T>
class EquationOfState : public ElastoPlasticModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(EquationOfState)

  EquationOfState(const T& youngs_modulus, const T& poissons_ratio,
                  const T& bulk, const T& gamma)
      : ElastoPlasticModel<T>(youngs_modulus, poissons_ratio) {
    bulk_ = bulk;
    gamma_ = gamma;
  }

  std::unique_ptr<ElastoPlasticModel<T>> Clone() const final {
    return std::make_unique<EquationOfState<T>>(*this);
  }

  bool IsLinearModel() const final { return false; }

  void CalcFEFromFtrial(const Matrix3<T>& F0, const Matrix3<T>& F_trial,
                        Matrix3<T>* FE) const final {
    DRAKE_ASSERT(FE != nullptr);
    *FE = F_trial;
    unused(F0);
  }

  T CalcStrainEnergyDensity(const Matrix3<T>& F0,
                            const Matrix3<T>& FE) const final {
    unused(F0);
    T J = FE(0, 0);
    using std::pow;
    return -bulk_ * (pow(J, 1.0 - gamma_) / (1.0 - gamma_) - J);
  }

  void CalcFirstPiolaStress(const Matrix3<T>& F0, const Matrix3<T>& FE,
                            Matrix3<T>* P) const final {
    unused(F0);
    T J = FE(0, 0);
    using std::pow;
    T scalar = -bulk_ * (pow(J, -gamma_) - 1.0);
    (*P) = Matrix3<T>::Identity() * scalar;
  }

  void CalcKirchhoffStress(const Matrix3<T>& F0, const Matrix3<T>& FE,
                           Matrix3<T>* tau) const final {
    Matrix3<T> P;
    CalcFirstPiolaStress(F0, FE, &P);
    (*tau) = P * FE.transpose();
  }

  void CalcFirstPiolaStressDerivative(const Matrix3<T>& F0,
                                      const Matrix3<T>& FE,
                                      Eigen::Matrix<T, 9, 9>* dPdF,
                                      bool project_pd = false) const final {
    unused(project_pd, F0);
    T J = FE(0, 0);
    using std::pow;
    T scalar = bulk_ * gamma_ * pow(J, -gamma_ - 1.0);
    (*dPdF) = scalar * Eigen::Matrix<T, 9, 9>::Identity();
  }

 private:

  T bulk_;
  T gamma_;
};

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
