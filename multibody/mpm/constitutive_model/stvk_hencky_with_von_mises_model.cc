#include "drake/multibody/mpm/constitutive_model/stvk_hencky_with_von_mises_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

template <typename T>
StvkHenckyWithVonMisesModel<T>::StvkHenckyWithVonMisesModel(const T& E,
                                                            const T& nu,
                                                            const T& tau_c)
    : ElastoPlasticModel<T>(E, nu), yield_stress_(tau_c) {
  DRAKE_ASSERT(tau_c > 0);
}

template <typename T>
T StvkHenckyWithVonMisesModel<T>::CalcStrainEnergyDensity(
    const Matrix3<T>& FE) const {
  // psi = μ tr((log Σ)^2) + 1/2 λ (tr(log Σ))^2
  StrainStressData strain_stress_data = ComputeStrainStressData(FE);

  return this->mu() * (strain_stress_data.log_sigma(0) *
                           strain_stress_data.log_sigma(0) +
                       strain_stress_data.log_sigma(1) *
                           strain_stress_data.log_sigma(1) +
                       strain_stress_data.log_sigma(2) *
                           strain_stress_data.log_sigma(2)) +
         0.5 * this->lambda() * strain_stress_data.log_sigma_trace *
             strain_stress_data.log_sigma_trace;
}

template <typename T>
void StvkHenckyWithVonMisesModel<T>::CalcFEFromFtrial(const Matrix3<T>& F_trial,
                                                      Matrix3<T>* FE) const {
  DRAKE_ASSERT(FE != nullptr);
  const StrainStressData strain_stress_data_trial =
      ComputeStrainStressData(F_trial);
  ApplyReturnMapping(strain_stress_data_trial, F_trial, FE);
}

template <typename T>
void StvkHenckyWithVonMisesModel<T>::CalcFirstPiolaStress(const Matrix3<T>& FE,
                                                          Matrix3<T>* P) const {
  // Here FE is *elastic* deformation gradient
  // P = U (2 μ Σ^{-1} (log Σ) + λ tr(log Σ) Σ^{-1}) V^T
  const StrainStressData strain_stress_data = ComputeStrainStressData(FE);
  Vector3<T> P_hat =
      2 * this->mu() * strain_stress_data.log_sigma +
      this->lambda() * strain_stress_data.log_sigma_trace * Vector3<T>::Ones();
  P_hat = strain_stress_data.one_over_sigma.asDiagonal() * P_hat;

  (*P) = strain_stress_data.U * P_hat.asDiagonal() *
         strain_stress_data.V.transpose();
}

template <typename T>
void StvkHenckyWithVonMisesModel<T>::CalcKirchhoffStress(
    const Matrix3<T>& FE, Matrix3<T>* tau) const {
  // Here FE is *elastic* deformation gradient
  // The Kirchhoff stress can be then written as
  // τ = U Σᵢ Uᵀ, where Σᵢ = 2μ log(σᵢ) + λ ∑ᵢ log(σᵢ).
  const StrainStressData strain_stress_data = ComputeStrainStressData(FE);
  Vector3<T> sigma_for_tau;
  for (int d = 0; d < 3; ++d) {
    sigma_for_tau(d) = 2 * this->mu() * strain_stress_data.log_sigma(d) +
                       this->lambda() * strain_stress_data.log_sigma_trace;
  }
  *tau = strain_stress_data.U * sigma_for_tau.asDiagonal() *
         strain_stress_data.U.transpose();
}

template <typename T>
void StvkHenckyWithVonMisesModel<T>::CalcFirstPiolaStressDerivative(
    const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const {
  const PsiSigmaDerivatives psi_derivatives = CalcPsiSigmaDerivative(FE);
  const StrainStressData ssd = ComputeStrainStressData(FE);
  for (int ij = 0; ij < 9; ++ij) {
    int j = ij / 3;
    int i = ij - j * 3;
    for (int rs = 0; rs <= ij; ++rs) {
      int s = rs / 3;
      int r = rs - s * 3;
      // computes the (i,j,r,s) entry of dPdF. It is also a symmetric matrix.
      // Following stvk_hencky_with_von_mises_model.md,
      // dPdF(i,j,r,s,) = Σₖₗₘₙ MₖₗₘₙUᵢₖVⱼₗUᵣₘVₛₙ.
      // 27 out of the 81 summands are non-zero, the corresponding k,l,m,n are
      // listed before each summand.
      // clang-format off
      (*dPdF)(ij, rs) = (*dPdF)(rs, ij) =
          // k,l,m,n = 0, 0, 0, 0
           psi_derivatives.psi_00 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 0, 0, 1, 1
         + psi_derivatives.psi_01 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 1) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 0, 0, 2, 2
         + psi_derivatives.psi_02 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 2) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 1, 1, 0, 0
         + psi_derivatives.psi_01 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 0) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 1, 1, 1, 1
         + psi_derivatives.psi_11 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 1, 1, 2, 2
         + psi_derivatives.psi_12 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 2) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 2, 2, 0, 0
         + psi_derivatives.psi_02 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 0) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 2, 2, 1, 1
         + psi_derivatives.psi_12 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 1) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 2, 2, 2, 2
         + psi_derivatives.psi_22 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 0, 1, 0, 1. B01(0,0)=M_0101
         + B01(psi_derivatives, 0, 0) * ssd.U(i, 0) * ssd.V(j, 1) * ssd.U(r, 0) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 0, 1, 1, 0. B01(0,1)=M_0110
         + B01(psi_derivatives, 0, 1) * ssd.U(i, 0) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 1, 0, 0, 1. B01(1,0)=M_1001=M_0110
         + B01(psi_derivatives, 1, 0) * ssd.U(i, 1) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 1, 0, 1, 0. B01(1,1)=M_1010=M_0101
         + B01(psi_derivatives, 1, 1) * ssd.U(i, 1) * ssd.V(j, 0) * ssd.U(r, 1) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 1, 2, 1, 2. B12(0,0)=M_1212
         + B12(psi_derivatives, 0, 0) * ssd.U(i, 1) * ssd.V(j, 2) * ssd.U(r, 1) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 1, 2, 2, 1. B12(0,1)=M_1221
         + B12(psi_derivatives, 0, 1) * ssd.U(i, 1) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 2, 1, 1, 2. B12(1,0)=M_2112=M_1221
         + B12(psi_derivatives, 1, 0) * ssd.U(i, 2) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 2, 1, 2, 1. B12(1,1)=M_2121=M_1212
         + B12(psi_derivatives, 1, 1) * ssd.U(i, 2) * ssd.V(j, 1) * ssd.U(r, 2) * ssd.V(s, 1) // NOLINT

          // k,l,m,n = 0, 2, 0, 2. B02(1,1)=M_0202
         + B02(psi_derivatives, 1, 1) * ssd.U(i, 0) * ssd.V(j, 2) * ssd.U(r, 0) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 0, 2, 2, 0. B02(1,0)=M_0220
         + B02(psi_derivatives, 1, 0) * ssd.U(i, 0) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 0) // NOLINT

          // k,l,m,n = 2, 0, 0, 2. B02(0,1)=M_2002=M_0220
         + B02(psi_derivatives, 0, 1) * ssd.U(i, 2) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 2) // NOLINT

          // k,l,m,n = 2, 0, 2, 0. B02(0,0)=M_2020=M_0202
         + B02(psi_derivatives, 0, 0) * ssd.U(i, 2) * ssd.V(j, 0) * ssd.U(r, 2) * ssd.V(s, 0);// NOLINT
      // clang-format off
    }
  }
}


template <typename T>
StvkHenckyWithVonMisesModel<T>::StrainStressData
StvkHenckyWithVonMisesModel<T>::ComputeStrainStressData(
    const Matrix3<T>& deformation_gradient) const {
  Eigen::JacobiSVD<Matrix3<T>> svd(deformation_gradient,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Vector3<T> sigma = svd.singularValues();
  DRAKE_ASSERT(sigma(0) > kEpsilon);
  DRAKE_ASSERT(sigma(1) > kEpsilon);
  DRAKE_ASSERT(sigma(2) > kEpsilon);
  const Vector3<T> one_over_sigma{1 / sigma(0), 1 / sigma(1), 1 / sigma(2)};
  const Vector3<T> log_sigma = sigma.array().log();
  const T log_sigma_trace = log_sigma.sum();
  // The deviatoric component of Kirchhoff stress in the principal frame is:
  // dev(τ) = τ - pJI = τ - 1/3tr(τ)I
  // In the principal frame: dev(τ) = 2μlog(σᵢ) - 2μ/3 ∑ᵢ log(σᵢ) =
  // 2μ[log(σᵢ)-1/3*∑ᵢ log(σᵢ)]
  const Vector3<T> deviatoric_tau =
      2 * this->mu() *
      (log_sigma - 1.0 / 3.0 * log_sigma_trace * Vector3<T>::Ones());
  return {.U = svd.matrixU(),
          .V = svd.matrixV(),
          .sigma = sigma,
          .one_over_sigma = one_over_sigma,
          .log_sigma = log_sigma,
          .log_sigma_trace = log_sigma_trace,
          .deviatoric_tau = deviatoric_tau};
}

template <typename T>
StvkHenckyWithVonMisesModel<T>::PsiSigmaDerivatives
StvkHenckyWithVonMisesModel<T>::CalcPsiSigmaDerivative(
    const Matrix3<T>& FE) const {
  const StrainStressData ssd = ComputeStrainStressData(FE);

  // dpsi_dsigma_i
  const T psi_0 = ssd.one_over_sigma(0) * (2 * this->mu() * ssd.log_sigma(0) +
                                   this->lambda() * ssd.log_sigma_trace);
  const T psi_1 = ssd.one_over_sigma(1) * (2 * this->mu() * ssd.log_sigma(1) +
                                   this->lambda() * ssd.log_sigma_trace);
  const T psi_2 = ssd.one_over_sigma(2) * (2 * this->mu() * ssd.log_sigma(2) +
                                   this->lambda() * ssd.log_sigma_trace);

  // d2psi_dsigma_i2
  const T lam_plus_2mu = 2 * this->mu() + this->lambda();
  const T psi_00 = ssd.one_over_sigma(0) * ssd.one_over_sigma(0) *
             (lam_plus_2mu * (1 - ssd.log_sigma(0)) -
              this->lambda() * (ssd.log_sigma(1) + ssd.log_sigma(2)));
  const T psi_11 = ssd.one_over_sigma(1) * ssd.one_over_sigma(1) *
             (lam_plus_2mu * (1 - ssd.log_sigma(1)) -
              this->lambda() * (ssd.log_sigma(0) + ssd.log_sigma(2)));
  const T psi_22 = ssd.one_over_sigma(2) * ssd.one_over_sigma(2) *
             (lam_plus_2mu * (1 - ssd.log_sigma(2)) -
              this->lambda() * (ssd.log_sigma(1) + ssd.log_sigma(0)));

  // d2psi_dsigma_i_dsigma_j
  const T psi_01 = this->lambda() * ssd.one_over_sigma(0)
              * ssd.one_over_sigma(1);
  const T psi_12 = this->lambda() * ssd.one_over_sigma(1)
              * ssd.one_over_sigma(2);
  const T psi_02 = this->lambda() * ssd.one_over_sigma(0)
              * ssd.one_over_sigma(2);

  // (psiA-psiB)/(sigmaA-sigmaB)
  using std::abs;

  const T m01 = -ssd.one_over_sigma(0) * ssd.one_over_sigma(1) *
          (this->lambda() * ssd.log_sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.sigma(0),
                                                           ssd.sigma(1)));

  const T m02 = -ssd.one_over_sigma(0) * ssd.one_over_sigma(2) *
          (this->lambda() * ssd.log_sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.sigma(0),
                                                           ssd.sigma(2)));

  const T m12 = -ssd.one_over_sigma(1) * ssd.one_over_sigma(2) *
          (this->lambda() * ssd.log_sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.sigma(1),
                                                           ssd.sigma(2)));

  // (psiA+psiB)/(sigmaA+sigmaB)
  const T p01 = (psi_0 + psi_1) /
          internal::ClampToEpsilon<T>(ssd.sigma(0) + ssd.sigma(1), kEpsilon);
  const T p02 = (psi_0 + psi_2) /
          internal::ClampToEpsilon<T>(ssd.sigma(0) + ssd.sigma(2), kEpsilon);
  const T p12 = (psi_1 + psi_2) /
          internal::ClampToEpsilon<T>(ssd.sigma(1) + ssd.sigma(2), kEpsilon);

  return {psi_0,  psi_1, psi_2, psi_00, psi_11, psi_22, psi_01, psi_02,
          psi_12, m01,   p01,   m02,    p02,    m12,    p12};
}

// The yield function is f(τ) = sqrt(3/2)‖ dev(τ) ‖ - τ_c.
// Plasticity is applied if f(τ) > 0.
template <typename T>
T StvkHenckyWithVonMisesModel<T>::ComputeYieldFunction(
    const Vector3<T>& deviatoric_tau) const {
  return kSqrt3Over2 * deviatoric_tau.norm() - yield_stress_;
}

// If the trial stress τ is in the yield surface f(τ) <= 0, plasticity is
// not applied.
// Otherwise, project the trial stress τ in the plastic flow direction
// df/dτ τ onto the yield surface f(τ) <= 0.
// Projection is done in the following way.
// Trial strain's projection onto yield surface in the principal frame
// is:
// ε_proj =  ε - Δγ νⁿ⁺¹,
//              where ν = νⁿ⁺¹ = 1/√(2/3)*dev(τⁿ⁺¹)/‖dev(τⁿ⁺¹)‖
//                            = 1/√(2/3)*dev(τ)/‖dev(τ)‖
//              Since dev(τ) and dev(τⁿ⁺¹) are in the same direction
// Taking the dot product of the associative flow rule with flow
// direction ν:
//     ν [dev(τⁿ⁺¹) - dev(τ)] = -2μ Δγ ‖ν‖²
// ==>         f(τⁿ⁺¹) - f(τ) = -3μ Δγ
// Since τⁿ⁺¹ is on the yield surface, f(τⁿ⁺¹) = 0.
// Δγ = f(τ)/(3μ)
template <typename T>
void StvkHenckyWithVonMisesModel<T>::ApplyReturnMapping(
    const StrainStressData& strain_stress_data_trial,
    const Matrix3<T>& F_trial, Matrix3<T>* FE) const {
  const T f_tau = ComputeYieldFunction(strain_stress_data_trial.deviatoric_tau);
  bool in_yield_surface = f_tau <= 0.0;
  if (!in_yield_surface) {
    // project trial strain onto yield surface in the principal frame
    const Vector3<T> nu =
        kSqrt3Over2 * strain_stress_data_trial.deviatoric_tau.normalized();
    const T delta_gamma = f_tau / (3.0 * this->mu());
    // Update the singular values of Hencky strain ε
    const Vector3<T> log_sigma_new =
        strain_stress_data_trial.log_sigma - delta_gamma * nu;
    // proj_F_hat = exp(ε_proj) in the principal frame
    const Vector3<T> proj_F_hat = (log_sigma_new).array().exp();
    // get the *elastic* deformation gradient from proj_F_hat
    *FE = strain_stress_data_trial.U * proj_F_hat.asDiagonal() *
               strain_stress_data_trial.V.transpose();
  } else {
    *FE = F_trial;
  }
}
template class StvkHenckyWithVonMisesModel<AutoDiffXd>;
template class StvkHenckyWithVonMisesModel<double>;

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
