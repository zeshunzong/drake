#include "drake/multibody/mpm/constitutive_model/stvk_hencky_with_von_mises_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

template <typename T>
StvkHenckyWithVonMisesModel2<T>::StvkHenckyWithVonMisesModel2(const T& E,
                                                              const T& nu,
                                                              const T& tau_c)
    : ElastoPlasticModel<T>(E, nu), yield_stress_(tau_c) {
  DRAKE_ASSERT(tau_c > 0);
}

template <typename T>
T StvkHenckyWithVonMisesModel2<T>::CalcStrainEnergyDensity(
    const Matrix3<T>& FE) const {
  // psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2
  StrainStressData strain_stress_data = ComputeStrainStressData(FE);

  return this->mu() * (strain_stress_data.log_Sigma(0) *
                           strain_stress_data.log_Sigma(0) +
                       strain_stress_data.log_Sigma(1) *
                           strain_stress_data.log_Sigma(1) +
                       strain_stress_data.log_Sigma(2) *
                           strain_stress_data.log_Sigma(2)) +
         0.5 * this->lambda() * strain_stress_data.log_Sigma_trace *
             strain_stress_data.log_Sigma_trace;
}

template <typename T>
void StvkHenckyWithVonMisesModel2<T>::CalcFEFromFtrial(
    Matrix3<T>* F_trial) const {
  DRAKE_ASSERT(F_trial != nullptr);
  const StrainStressData strain_stress_data_trial =
      ComputeStrainStressData(*F_trial);
  ApplyReturnMapping(strain_stress_data_trial, F_trial);
}

template <typename T>
void StvkHenckyWithVonMisesModel2<T>::CalcFirstPiolaStress(
    const Matrix3<T>& FE, Matrix3<T>* P) const {
  // Here FE is *elastic* deformation gradient
  // P = U (2 mu S^{-1} (log S) + lambda tr(log S) S^{-1}) V^T
  const StrainStressData strain_stress_data = ComputeStrainStressData(FE);
  Vector3<T> P_hat =
      2 * this->mu() * strain_stress_data.log_Sigma +
      this->lambda() * strain_stress_data.log_Sigma_trace * Vector3<T>::Ones();
  P_hat = strain_stress_data.OneOverSigma.asDiagonal() * P_hat;

  (*P) = strain_stress_data.U * P_hat.asDiagonal() *
         strain_stress_data.V.transpose();
}

template <typename T>
void StvkHenckyWithVonMisesModel2<T>::CalcKirchhoffStress(
    const Matrix3<T>& FE, Matrix3<T>* tau) const {
  // Here FE is *elastic* deformation gradient

  // The Kirchhoff stress can be then written as
  // τ = U Σᵢ Uᵀ, where U are left singular vectors of the elastic deformation
  // gradient FE Σᵢ = 2μ log(σᵢ) + λ ∑ᵢ log(σᵢ).
  const StrainStressData strain_stress_data = ComputeStrainStressData(FE);
  Vector3<T> sigma_for_tau;
  for (int d = 0; d < 3; ++d) {
    sigma_for_tau(d) = 2 * this->mu() * strain_stress_data.log_Sigma(d) +
                       this->lambda() * strain_stress_data.log_Sigma_trace;
  }
  *tau = strain_stress_data.U * sigma_for_tau.asDiagonal() *
         strain_stress_data.U.transpose();
}

template <typename T>
void StvkHenckyWithVonMisesModel2<T>::CalcFirstPiolaStressDerivative(
    const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const {
  const PsiSigmaDerivatives psi_derivatives = CalcPsiSigmaDerivative(FE);
  const StrainStressData ssd = ComputeStrainStressData(FE);
  for (int ij = 0; ij < 9; ++ij) {
    int j = ij / 3;
    int i = ij - j * 3;
    for (int rs = 0; rs <= ij; ++rs) {
      int s = rs / 3;
      int r = rs - s * 3;
      // writes the (i,j,r,s) entry of dPdF. It is also a symmetric matrix.
      // Following stvk_hencky_with_von_mises_model.md,
      // dPdF(i,j,r,s,) = Σₖₗₘₙ MₖₗₘₙUᵢₖVⱼₗUᵣₘVₛₙ
      // 27 out of the 81 summands are non-zero. 
      // 0000 0011 0022 1100 1111 1122 2200 2211 2222


      (*dPdF)(ij, rs) = (*dPdF)(rs, ij) =
           psi_derivatives.psi_00 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 0) // 0000

         + psi_derivatives.psi_01 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 1) * ssd.V(s, 1) // 0011

         + psi_derivatives.psi_02 * ssd.U(i, 0) * ssd.V(j, 0) * ssd.U(r, 2) * ssd.V(s, 2) // 0022

         + psi_derivatives.psi_01 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 0) * ssd.V(s, 0) // 1100

         + psi_derivatives.psi_11 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 1) // 1111

         + psi_derivatives.psi_12 * ssd.U(i, 1) * ssd.V(j, 1) * ssd.U(r, 2) * ssd.V(s, 2) // 1122

         + psi_derivatives.psi_02 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 0) * ssd.V(s, 0) // 2200

         + psi_derivatives.psi_12 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 1) * ssd.V(s, 1) // 2211
         
         + psi_derivatives.psi_22 * ssd.U(i, 2) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 2) // 2222
    
         + B01(psi_derivatives, 0, 0) * ssd.U(i, 0) * ssd.V(j, 1) * ssd.U(r, 0) * ssd.V(s, 1) // 0101, B01(0,0)=M_0101
         
         + B01(psi_derivatives, 0, 1) * ssd.U(i, 0) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 0) // 0110, B01(0,1)=M_0110
         
         + B01(psi_derivatives, 1, 0) * ssd.U(i, 1) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 1) // 1001, B01(1,0)=M_1001=M_0110
         
         + B01(psi_derivatives, 1, 1) * ssd.U(i, 1) * ssd.V(j, 0) * ssd.U(r, 1) * ssd.V(s, 0) // 1010, B01(1,1)=M_1010=M_0101
         
         + B12(psi_derivatives, 0, 0) * ssd.U(i, 1) * ssd.V(j, 2) * ssd.U(r, 1) * ssd.V(s, 2) // 1212, B12(0,0)=M_1212
         
         + B12(psi_derivatives, 0, 1) * ssd.U(i, 1) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 1) // 1221, B12(0,1)=M_1221
         
         + B12(psi_derivatives, 1, 0) * ssd.U(i, 2) * ssd.V(j, 1) * ssd.U(r, 1) * ssd.V(s, 2) // 2112, B12(1,0)=M_2112=M_1221
         
         + B12(psi_derivatives, 1, 1) * ssd.U(i, 2) * ssd.V(j, 1) * ssd.U(r, 2) * ssd.V(s, 1) // 2121, B12(1,1)=M_2121=M_1212
         
         + B20(psi_derivatives, 1, 1) * ssd.U(i, 0) * ssd.V(j, 2) * ssd.U(r, 0) * ssd.V(s, 2) // 0202, B20(1,1)=M_0202
         
         + B20(psi_derivatives, 1, 0) * ssd.U(i, 0) * ssd.V(j, 2) * ssd.U(r, 2) * ssd.V(s, 0) // 0220, B20(1,0)=M_0220
         
         + B20(psi_derivatives, 0, 1) * ssd.U(i, 2) * ssd.V(j, 0) * ssd.U(r, 0) * ssd.V(s, 2) // 2002, B20(0,1)=M_2002=M_0220
         
         + B20(psi_derivatives, 0, 0) * ssd.U(i, 2) * ssd.V(j, 0) * ssd.U(r, 2) * ssd.V(s, 0);// 2020, B20(0,0)=M_2020=M_0202
    }
  }
}

// TODO: remove this function
template <typename T>
void StvkHenckyWithVonMisesModel2<
    T>::UpdateDeformationGradientAndCalcKirchhoffStress(Matrix3<T>* tau,
                                                        Matrix3<T>* FE_trial)
    const {
  DRAKE_ASSERT(tau != nullptr);
  DRAKE_ASSERT(FE_trial != nullptr);
}

template <typename T>
StvkHenckyWithVonMisesModel2<T>::StrainStressData
StvkHenckyWithVonMisesModel2<T>::ComputeStrainStressData(
    const Matrix3<T>& F_trial) const {
  Eigen::JacobiSVD<Matrix3<T>> svd(F_trial,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Matrix3<T>& U = svd.matrixU();
  const Matrix3<T>& V = svd.matrixV();
  Vector3<T> Sigma = svd.singularValues();
  Vector3<T> OneOverSigma{1 / Sigma(0), 1 / Sigma(1), 1 / Sigma(2)};
  Vector3<T> log_Sigma = Sigma.array().log();
  T log_Sigma_trace = log_Sigma.sum();

  // The deviatoric component of Kirchhoff stress in the principal frame is:
  // dev(τ) = τ - pJI = τ - 1/3tr(τ)I
  // In the principal frame: dev(τ) = 2μlog(σᵢ) - 2μ/3 ∑ᵢ log(σᵢ) =
  // 2μ[log(σᵢ)-1/3*∑ᵢ log(σᵢ)]
  Vector3<T> deviatoric_tau =
      2 * this->mu() *
      (log_Sigma - 1.0 / 3.0 * log_Sigma_trace * Vector3<T>::Ones());
  return {
      U, V, Sigma, OneOverSigma, log_Sigma, log_Sigma_trace, deviatoric_tau};
}

template <typename T>
StvkHenckyWithVonMisesModel2<T>::PsiSigmaDerivatives
StvkHenckyWithVonMisesModel2<T>::CalcPsiSigmaDerivative(
    const Matrix3<T>& FE) const {
  const StrainStressData ssd = ComputeStrainStressData(FE);

  // dpsi_dsigma_i
  T psi_0 = ssd.OneOverSigma(0) * (2 * this->mu() * ssd.log_Sigma(0) +
                                   this->lambda() * ssd.log_Sigma_trace);
  T psi_1 = ssd.OneOverSigma(1) * (2 * this->mu() * ssd.log_Sigma(1) +
                                   this->lambda() * ssd.log_Sigma_trace);
  T psi_2 = ssd.OneOverSigma(2) * (2 * this->mu() * ssd.log_Sigma(2) +
                                   this->lambda() * ssd.log_Sigma_trace);

  // d2psi_dsigma_i2
  T lam_plus_2mu = 2 * this->mu() + this->lambda();
  T psi_00 = ssd.OneOverSigma(0) * ssd.OneOverSigma(0) *
             (lam_plus_2mu * (1 - ssd.log_Sigma(0)) -
              this->lambda() * (ssd.log_Sigma(1) + ssd.log_Sigma(2)));
  T psi_11 = ssd.OneOverSigma(1) * ssd.OneOverSigma(1) *
             (lam_plus_2mu * (1 - ssd.log_Sigma(1)) -
              this->lambda() * (ssd.log_Sigma(0) + ssd.log_Sigma(2)));
  T psi_22 = ssd.OneOverSigma(2) * ssd.OneOverSigma(2) *
             (lam_plus_2mu * (1 - ssd.log_Sigma(2)) -
              this->lambda() * (ssd.log_Sigma(1) + ssd.log_Sigma(0)));

  // d2psi_dsigma_i_dsigma_j
  T psi_01 = this->lambda() * ssd.OneOverSigma(0) * ssd.OneOverSigma(1);
  T psi_12 = this->lambda() * ssd.OneOverSigma(1) * ssd.OneOverSigma(2);
  T psi_02 = this->lambda() * ssd.OneOverSigma(0) * ssd.OneOverSigma(2);

  // (psiA-psiB)/(sigmaA-sigmaB)
  using std::abs;

  T m01 = -ssd.OneOverSigma(0) * ssd.OneOverSigma(1) *
          (this->lambda() * ssd.log_Sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.Sigma(0),
                                                           ssd.Sigma(1)));

  T m02 = -ssd.OneOverSigma(0) * ssd.OneOverSigma(2) *
          (this->lambda() * ssd.log_Sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.Sigma(0),
                                                           ssd.Sigma(2)));

  T m12 = -ssd.OneOverSigma(1) * ssd.OneOverSigma(2) *
          (this->lambda() * ssd.log_Sigma_trace +
           2 * this->mu() *
               internal::CalcXLogYMinusYLogXOverXMinusY<T>(ssd.Sigma(1),
                                                           ssd.Sigma(2)));

  // (psiA+psiB)/(sigmaA+sigmaB)
  T p01 = (psi_0 + psi_1) /
          internal::ClampToEpsilon<T>(ssd.Sigma(0) + ssd.Sigma(1), kEpsilon);
  T p02 = (psi_0 + psi_2) /
          internal::ClampToEpsilon<T>(ssd.Sigma(0) + ssd.Sigma(2), kEpsilon);
  T p12 = (psi_1 + psi_2) /
          internal::ClampToEpsilon<T>(ssd.Sigma(1) + ssd.Sigma(2), kEpsilon);

  return {psi_0,  psi_1, psi_2, psi_00, psi_11, psi_22, psi_01, psi_02,
          psi_12, m01,   p01,   m02,    p02,    m12,    p12};
}

// The yield function is f(τ) = sqrt(3/2)‖ dev(τ) ‖ - τ_c.
// Plasticity is applied if f(τ) > 0.
template <typename T>
T StvkHenckyWithVonMisesModel2<T>::ComputeYieldFunction(
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
//     ν [dev(τⁿ⁺¹) - dev(τ)] = -2μ Δγ ‖ν‖
// ==>         f(τⁿ⁺¹) - f(τ) = -3μ Δγ
// Since f(τⁿ⁺¹) = ‖ τⁿ⁺¹ ‖ - τ_c = 0, τⁿ⁺¹ is on the yield surface
// Δγ = f(τ)/(3μ)
template <typename T>
void StvkHenckyWithVonMisesModel2<T>::ApplyReturnMapping(
    const StrainStressData& strain_stress_data_trial,
    Matrix3<T>* F_trial) const {
  T f_tau = ComputeYieldFunction(strain_stress_data_trial.deviatoric_tau);
  bool in_yield_surface = f_tau <= 0.0;
  if (!in_yield_surface) {
    // project trial strain onto yield surface in the principal frame
    Vector3<T> nu = kSqrt3Over2 * strain_stress_data_trial.deviatoric_tau /
                    strain_stress_data_trial.deviatoric_tau.norm();
    T delta_gamma = f_tau / (3.0 * this->mu());
    // Update the singular values of Hencky strain ε
    Vector3<T> log_Sigma_new =
        strain_stress_data_trial.log_Sigma - delta_gamma * nu;
    // F_proj = exp(ε_proj) in the principal frame
    Vector3<T> proj_F_hat = (log_Sigma_new).array().exp();
    // New elastic deformation gradient projected to the yield surface
    // proj(Fᴱ). This is essentially *elastic* deformation gradient FE
    *F_trial = strain_stress_data_trial.U * proj_F_hat.asDiagonal() *
               strain_stress_data_trial.V.transpose();
  }
}
template class StvkHenckyWithVonMisesModel2<AutoDiffXd>;
template class StvkHenckyWithVonMisesModel2<double>;

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
