#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

template <typename T>
CorotatedElasticModel<T>::CorotatedElasticModel(const T& youngs_modulus,
                                                const T& poissons_ratio)
    : ElastoPlasticModel<T>(youngs_modulus, poissons_ratio) {}

template <typename T>
T CorotatedElasticModel<T>::CalcStrainEnergyDensity(
    const Matrix3<T>& FE) const {
  Eigen::JacobiSVD<Matrix3<T>> svd(FE);
  Vector3<T> sigma = svd.singularValues();
  T J = sigma(0) * sigma(1) * sigma(2);
  return this->mu() * ((sigma(0) - 1.0) * (sigma(0) - 1.0) +
                       (sigma(1) - 1.0) * (sigma(1) - 1.0) +
                       (sigma(2) - 1.0) * (sigma(2) - 1.0)) +
         this->lambda() / 2.0 * (J - 1) * (J - 1);
}

template <typename T>
void CorotatedElasticModel<T>::CalcFEFromFtrial(const Matrix3<T>& F_trial,
                                                Matrix3<T>* FE) const {
  DRAKE_ASSERT(FE != nullptr);
  *FE = F_trial;
}

template <typename T>
void CorotatedElasticModel<T>::CalcFirstPiolaStress(const Matrix3<T>& FE,
                                                    Matrix3<T>* P) const {
  Matrix3<T> R, S, JFinvT;
  fem::internal::PolarDecompose<T>(FE, &R, &S);
  fem::internal::CalcCofactorMatrix<T>(FE, &JFinvT);
  (*P) = 2.0 * this->mu() * (FE - R) +
         this->lambda() * (FE.determinant() - 1.0) * JFinvT;
}

template <typename T>
void CorotatedElasticModel<T>::CalcKirchhoffStress(const Matrix3<T>& FE,
                                                   Matrix3<T>* tau) const {
  Matrix3<T> R, S;
  T J = FE.determinant();
  fem::internal::PolarDecompose<T>(FE, &R, &S);
  *tau = 2.0 * this->mu() * (FE - R) * FE.transpose() +
         this->lambda() * (J - 1.0) * J * Matrix3<T>::Identity();
}

template <typename T>
void CorotatedElasticModel<T>::CalcFirstPiolaStressDerivative(
    const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const {
  Matrix3<T> R, S, JFinvT;
  fem::internal::PolarDecompose<T>(FE, &R, &S);
  fem::internal::CalcCofactorMatrix<T>(FE, &JFinvT);
  const Vector<T, 3 * 3>& flat_JFinvT =
      Eigen::Map<const Vector<T, 3 * 3>>(JFinvT.data(), 3 * 3);
  // The contribution from derivatives of Jm1.
  (*dPdF) = this->lambda() * flat_JFinvT * flat_JFinvT.transpose();
  // The contribution from derivatives of F.
  (*dPdF).diagonal().array() += 2.0 * this->mu();
  // The contribution from derivatives of R.
  fem::internal::AddScaledRotationalDerivative<T>(R, S, -2.0 * this->mu(),
                                                  dPdF);
  // The contribution from derivatives of JFinvT.
  fem::internal::AddScaledCofactorMatrixDerivative<T>(
      FE, this->lambda() * (FE.determinant() - 1.0), dPdF);
}

template class CorotatedElasticModel<AutoDiffXd>;
template class CorotatedElasticModel<double>;

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
