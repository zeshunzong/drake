#pragma once
#include <memory>

#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

// TODO(zeshunzong): write doc for this class
template <typename T>
class LinearCorotatedWithPlasticity : public ElastoPlasticModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(LinearCorotatedWithPlasticity)

  LinearCorotatedWithPlasticity(const T& E, const T& nu, const T& yield_stress)
      : ElastoPlasticModel<T>(E, nu), yield_stress_(yield_stress) {
    DRAKE_ASSERT(yield_stress > 0);
  }

  std::unique_ptr<ElastoPlasticModel<T>> Clone() const final {
    return std::make_unique<LinearCorotatedWithPlasticity<T>>(*this);
  }

  bool IsLinearModel() const final { return true; }

  void CalcFEFromFtrial(const Matrix3<T>& F0, const Matrix3<T>& F_trial,
                        Matrix3<T>* FE) const final {
    Matrix3<T> R0;
    Matrix3<T> unused;
    fem::internal::PolarDecompose<T>(F0, &R0, &unused);
    // Matrix3<T> S = 0.5 * (R0.transpose() * F_trial + F_trial.transpose() * R0);
    Matrix3<T> S = 0.5 * (F_trial * R0.transpose() + R0 * F_trial.transpose());
    Eigen::JacobiSVD<Matrix3<T>> svd(S,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3<T> sigma = svd.singularValues();
    internal::ProjectToSkewedCylinder<T>(yield_stress_ / (2.0 * this->mu()),
                                      &sigma);

    Matrix3<T> new_S =
        svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();

    *FE = F_trial;
    internal::SolveForNewF2<T>(new_S, R0, FE);    
  }

  struct StrainData {
    Matrix3<T> R0;
    Matrix3<T> strain;
    T trace_strain{};
  };

  T CalcStrainEnergyDensity(const Matrix3<T>& F0,
                            const Matrix3<T>& FE) const final {
    StrainData data = ComputeStrainData(F0, FE);
    return this->mu() * data.strain.squaredNorm() +
           0.5 * this->lambda() * data.trace_strain * data.trace_strain;
  }

  void CalcFirstPiolaStress(const Matrix3<T>& F0, const Matrix3<T>& FE,
                            Matrix3<T>* P) const final {
    StrainData data = ComputeStrainData(F0, FE);
    (*P) = 2.0 * this->mu() * data.R0 * data.strain +
           this->lambda() * data.trace_strain * data.R0;
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
    if (project_pd) {
      throw std::logic_error("This model is linear");
    }
    StrainData data = ComputeStrainData(F0, FE);

    /* Add in μ * δₐᵢδⱼᵦ. */
    (*dPdF) = this->mu() * Eigen::Matrix<T, 9, 9>::Identity();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int alpha = 0; alpha < 3; ++alpha) {
          for (int beta = 0; beta < 3; ++beta) {
            /* Add in  μ *  Rᵢᵦ Rₐⱼ +   λ * Rₐᵦ * Rᵢⱼ. */
            (*dPdF)(3 * j + i, 3 * beta + alpha) +=
                this->mu() * data.R0(i, beta) * data.R0(alpha, j) +
                this->lambda() * data.R0(alpha, beta) * data.R0(i, j);
          }
        }
      }
    }
  }

 private:
  StrainData ComputeStrainData(const Matrix3<T>& F0,
                               const Matrix3<T>& FE) const {
    Matrix3<T> R0;
    Matrix3<T> strain;
    Matrix3<T> unused_S;
    fem::internal::PolarDecompose<T>(F0, &R0, &unused_S);
    const Matrix3<T> corotated_F = R0.transpose() * FE;
    strain =
        0.5 * (corotated_F + corotated_F.transpose()) - Matrix3<T>::Identity();
    T trace_strain = strain.trace();
    return {R0, strain, trace_strain};
  }

  T yield_stress_{};
};

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
