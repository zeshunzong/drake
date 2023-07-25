#include "drake/multibody/mpm/CorotatedElasticModel.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
CorotatedElasticModel<T>::CorotatedElasticModel(): ElastoPlasticModel<T>() {}

template <typename T>
CorotatedElasticModel<T>::CorotatedElasticModel(T E, T nu):
                                                ElastoPlasticModel<T>(E, nu) {}

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
void CorotatedElasticModel<T>::UpdateDeformationGradientAndCalcKirchhoffStress(
                        Matrix3<T>* tau, Matrix3<T>* FE) const {
    Matrix3<T> R, S;
    T J = FE->determinant();
    fem::internal::PolarDecompose<T>(*FE, &R, &S);
    *tau = 2.0*ElastoPlasticModel<T>::mu_*(*FE-R)*FE->transpose()
         + ElastoPlasticModel<T>::lambda_*(J-1.0)*J*Matrix3<T>::Identity();
}

template class CorotatedElasticModel<double>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
