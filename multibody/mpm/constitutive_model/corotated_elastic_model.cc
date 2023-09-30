#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

template <typename T>
CorotatedElasticModel<T>::CorotatedElasticModel(): ElastoPlasticModel<T>() {}

template <typename T>
CorotatedElasticModel<T>::CorotatedElasticModel(const T& youngs_modulus, const T& poissons_ratio):
                                                ElastoPlasticModel<T>(youngs_modulus, poissons_ratio) {}

template <typename T>
T CorotatedElasticModel<T>::CalcStrainEnergyDensity(const Matrix3<T>& FE)
                                                                      const {
     Eigen::JacobiSVD<Matrix3<T>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3<T> sigma    = svd.singularValues();
    T J = sigma(0)*sigma(1)*sigma(2);
    return ElastoPlasticModel<T>::mu_*((sigma(0)-1.0)*(sigma(0)-1.0)
               +(sigma(1)-1.0)*(sigma(1)-1.0)
               +(sigma(2)-1.0)*(sigma(2)-1.0))
         + ElastoPlasticModel<T>::lambda_/2.0*(J-1)*(J-1);
}

template <typename T>
void CorotatedElasticModel<T>::CalcFEFromFtrial(Matrix3<T>* F_trial) const {
      DRAKE_ASSERT(F_trial != nullptr);
}

template <typename T>
void CorotatedElasticModel<T>::CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const {
      const T Jm1 = this->ComputeJMinusOne(FE);
      Matrix3<T> R, S, JFinvT;
      this->ComputePolarDecomp(FE, &R, &S);
      this->ComputeJFinvT(FE, &JFinvT);
      (*P) = 2.0 * this->mu_ * (FE - R) + this->lambda_ * Jm1 * JFinvT;
}

template <typename T>
void CorotatedElasticModel<T>::CalcKirchhoffStress(const Matrix3<T>& FE, Matrix3<T>* tau) const {
      Matrix3<T> R, S;
    T J = FE.determinant();
    fem::internal::PolarDecompose<T>(FE, &R, &S);
    *tau = 2.0*ElastoPlasticModel<T>::mu_*(FE-R)*FE.transpose()
         + ElastoPlasticModel<T>::lambda_*(J-1.0)*J*Matrix3<T>::Identity();
}

template <typename T>
void CorotatedElasticModel<T>::CalcFirstPiolaStressDerivative(const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const {
      dPdF->setZero();
      const T Jm1 = this->ComputeJMinusOne(FE);
      Matrix3<T> R, S, JFinvT;
      this->ComputePolarDecomp(FE, &R, &S);
      this->ComputeJFinvT(FE, &JFinvT);
      const Vector<T, 3 * 3>& flat_JFinvT =
        Eigen::Map<const Vector<T, 3 * 3>>(JFinvT.data(), 3 * 3);
      /* The contribution from derivatives of Jm1. */
      (*dPdF) = this->lambda_ * flat_JFinvT * flat_JFinvT.transpose();
      /* The contribution from derivatives of F. */
      (*dPdF).diagonal().array() += 2.0 * this->mu_;
      /* The contribution from derivatives of R. */
      fem::internal::AddScaledRotationalDerivative<T>(R, S, -2.0 * this->mu_, dPdF);
      /* The contribution from derivatives of JFinvT. */
      fem::internal::AddScaledCofactorMatrixDerivative<T>(FE, this->lambda_ * Jm1,
                                                   dPdF);
}

template <typename T>
void CorotatedElasticModel<T>::UpdateDeformationGradientAndCalcKirchhoffStress(
                        Matrix3<T>* tau, Matrix3<T>* FE) const {
    Matrix3<T> R, S;
    T J = FE->determinant();
    fem::internal::PolarDecompose<T>(*FE, &R, &S);
    *tau = 2.0*ElastoPlasticModel<T>::mu_*(*FE-R)*FE->transpose()
         + ElastoPlasticModel<T>::lambda_*(J-1.0)*J*Matrix3<T>::Identity();
}
template class CorotatedElasticModel<AutoDiffXd>;
template class CorotatedElasticModel<double>;

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
