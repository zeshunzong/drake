#include "drake/multibody/mpm/CorotatedElasticModel.h"

namespace drake {
namespace multibody {
namespace mpm {

CorotatedElasticModel::CorotatedElasticModel(): ElastoPlasticModel() {}

CorotatedElasticModel::CorotatedElasticModel(double E, double nu):
                                                ElastoPlasticModel(E, nu) {}

double CorotatedElasticModel::CalcStrainEnergyDensity(const Matrix3<double>& FE)
                                                                      const {
     Eigen::JacobiSVD<Matrix3<double>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3<double> sigma    = svd.singularValues();
    double J = sigma(0)*sigma(1)*sigma(2);
    return mu_*((sigma(0)-1.0)*(sigma(0)-1.0)
               +(sigma(1)-1.0)*(sigma(1)-1.0)
               +(sigma(2)-1.0)*(sigma(2)-1.0))
         + lambda_/2.0*(J-1)*(J-1);
}

void CorotatedElasticModel::UpdateDeformationGradientAndCalcKirchhoffStress(
                        Matrix3<double>* tau, Matrix3<double>* FE) const {
    Matrix3<double> R, S;
    double J = FE->determinant();
    fem::internal::PolarDecompose<double>(*FE, &R, &S);
    *tau = 2.0*mu_*(*FE-R)*FE->transpose()
         + lambda_*(J-1.0)*J*Matrix3<double>::Identity();
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
