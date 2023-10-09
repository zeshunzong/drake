#pragma once

#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

// A implementation of Saint-Venant Kirchhoff model, but replace the left
// Cauchy Green strain with the Hencky strain
// https://dl.acm.org/doi/abs/10.1145/2897824.2925906
// The formula of Kirchhoff stress can be found in Section 6.3,
// and von Mises plasticity model, described in Bonet&Wood (BW)
// https://www.klancek.si/sites/default/files/datoteke/files/
// bonet-woodnonlinearcontinuummechanics2ndedition.pdf
// The return mapping algorithm can be found in Box 7.1.
// Assume the constitutive model we are using is the Saint-Venant Kirchhoff
// constitutive model with Hencky strain. A plasticity model composed
// of a yield surface {τ: f(τ) <= 0}, an implicit surface describing the
// elastic/plastic limit. And an associate flow rule, an ODE relating the
// plastic deformation rate and stress, l_p = dγ/dt df(τ)/dτ, where l_p is the
// plastic rate of deformation and dγ/dt is called the plastic multiplier.
// We first outline the general procedure of applying plasticity
// 1. Update the elastic deformation gradient Fᴱ in the G2P update, compute the
//    corresponding Kirchhoff stress τ, and we call it the trial stress.
// 2. Project the trial stress to the yield surface, get proj(τ), if the trial
//    stress is outside the yield surface
// 3. Recover the projected deformation gradient proj(Fᴱ) by proj(τ) by
//    inverting the constitutive relation
//
// In particular, the Von Mises yield function is (BW 7.20a,b)
//      f(τ) = √(3/2) ‖ dev(τ) ‖_F - τ_c,
//                                τ_c is the maximum allowed tensile strength
// The associate flow rule for Von Mises model can be written as (BW 7.45a,b)
//      ln(Σⁿ⁺¹) - ln(Σ) = -Δγ νⁿ⁺¹
// Given the SVD of elastic deformation gradient Fᴱ = U Σ Vᵀ,
// the Hencky strain             ε = 1/2ln(Fᴱ Fᴱ^T) = U ln(Σ) Uᵀ
// the energy density            ψ = U Σ⁻¹(μln(Σ)²+1/2*λtr(ln(Σ))²I) Vᵀ
// the (trial) Kirchhoff stress  τ = U (2μln(Σ)+λtr(ln(Σ))I) Uᵀ
// the deviatoric component of trial stress is
//                  dev(τ) = τ - 1/3*tr(τ)I = U (2μln(Σ)+2/3*μ tr(ln(Σ))I) Uᵀ
// The we can rewrite the associate flow rule with (BW 7.53)
//       dev(τⁿ⁺¹) - dev(τ) = -2μ Δγ νⁿ⁺¹
// τⁿ⁺¹ denotes the projection of the trial stress: proj(τ)
// One key observation is that the flow direction νⁿ⁺¹ is has the same direction
// as the deviatoric components of updated Kirchhoff stress (BW 7.40)
//       νⁿ⁺¹ = df(τⁿ⁺¹)/dτ = dev(τⁿ⁺¹)/(√(3/2) ‖ dev(τⁿ⁺¹) ‖_F)
// As a result, the deviatoric components of updated and trial Kirchhoff stress
// are in the same direction, i.e.
//      dev(τⁿ⁺¹) = k dev(τ)
//                 k: f(τⁿ⁺¹) = 0 on the yield surface
// Solving for k gives us the prjected states. The above derivation leads to the
// same algorithm described in (BW Box 7.1), without hardening. We assume
// no hardening in the plastic model, so the yield surface (function) doesn't
// change with respect to time. This is equivalent to the formulation in
// (BW Box 7.1) with H = 0.
template <typename T>
class StvkHenckyWithVonMisesModel2 : public ElastoPlasticModel<T> {
 public:
  /**
   * Yield stress is the minimum stress at which the material undergoes plastic
   * deformation
   * @pre yield_stress > 0
   */
  StvkHenckyWithVonMisesModel2(const T& E, const T& nu, const T& yield_stress);


  virtual std::unique_ptr<ElastoPlasticModel<T>> Clone() const {
    return std::make_unique<StvkHenckyWithVonMisesModel2<T>>(*this);
  }

  T CalcStrainEnergyDensity(const Matrix3<T>& FE) const final;

  void CalcFEFromFtrial(Matrix3<T>* F_trial) const final;

  void CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const final;

  void CalcKirchhoffStress(const Matrix3<T>& FE, Matrix3<T>* tau) const final;

  void CalcFirstPiolaStressDerivative(const Matrix3<T>& FE,
                                      Eigen::Matrix<T, 9, 9>* dPdF) const final;

  void UpdateDeformationGradientAndCalcKirchhoffStress(
      Matrix3<T>* tau,
      Matrix3<T>* trial_elastic_deformation_gradient) const final;

 private:
  // Given the Hencky strain ε, assume the SVD of it is U Σ Vᵀ,
  // We store U in `U`, V in `V`, eps_hat in `Σ`, and tr(Σ) in `tr_eps`
  struct StrainData {
    Matrix3<T> U;
    Matrix3<T> V;
    Vector3<T> eps_hat;
    T tr_eps;
    Vector3<T> sigma;
    Vector3<T> sigma_inverse;
  };
  
  // Given deformation gradient F, stores the following results
  // U, V: from svd such that F = U Σ Vᵀ, Matrix3
  // Sigma: diag(Σ), Vector3
  // OneOverSigma: 1./diag(Σ), Vector3
  // log_Sigma: log(diag(Σ)), Vector3
  // log_Sigma_trace: sum(log(diag(Σ))), scalar
  // deviatoric_tau: the deviatoric component of τ in the principal frame, Vector3
  // N.B.: user should pay attention to whether the data is computed from
  // F_trial or FE
  
  struct StrainStressData {
    Matrix3<T> U;
    Matrix3<T> V;
    Vector3<T> Sigma;
    Vector3<T> OneOverSigma;
    Vector3<T> log_Sigma;
    T log_Sigma_trace;
    Vector3<T> deviatoric_tau;
  };

  StrainStressData ComputeStrainStressData(const Matrix3<T>& F_trial) const;

  // derivatives of energy density psi w.r.t singular values sigma0, sigma1, and
  // sigma2
  struct PsiSigmaDerivatives {
    // Naming rule:
    // psi_i  := dψ/dσᵢ
    // psi_ij := dψ²/dσᵢdσⱼ
    T psi_0;   // d_psi_d_sigma0
    T psi_1;   // d_psi_d_sigma1
    T psi_2;   // d_psi_d_sigma2
    T psi_00;  // d^2_psi_d_sigma0_d_sigma0
    T psi_11;  // d^2_psi_d_sigma1_d_sigma1
    T psi_22;  // d^2_psi_d_sigma2_d_sigma2
    T psi_01;  // d^2_psi_d_sigma0_d_sigma1
    T psi_02;  // d^2_psi_d_sigma0_d_sigma2
    T psi_12;  // d^2_psi_d_sigma1_d_sigma2


    T m01;    // (psi_0-psi_1)/(sigma0-sigma1), usually can be computed robustly
    T p01;    // (psi_0+psi_1)/(sigma0+sigma1), need to clamp bottom with 1e-6
    T m02;    // (psi_0-psi_2)/(sigma0-sigma2), usually can be computed robustly
    T p02;    // (psi_0+psi_2)/(sigma0+sigma2), need to clamp bottom with 1e-6
    T m12;    // (psi_1-psi_2)/(sigma1-sigma2), usually can be computed robustly
    T p12;    // (psi_1+psi_2)/(sigma1+sigma2), need to clamp bottom with 1e-6

    Vector3<T> dpsi_dsigma_i;
  };
  
  // Returns the (i,j)'s entry of Matrix B01.
  // @pre i = 0 or 1
  // @pre j = 0 or 1
  T B01(const PsiSigmaDerivatives& psi_derivatives, int i, int j) const {
    return (i == j ? psi_derivatives.m01 + psi_derivatives.p01
                   : psi_derivatives.m01 - psi_derivatives.p01) *
           0.5;
  }

  // Returns the (i,j)'s entry of Matrix B12.
  // @pre i = 0 or 1
  // @pre j = 0 or 1
  T B12(const PsiSigmaDerivatives& psi_derivatives, int i, int j) const {
    return (i == j ? psi_derivatives.m12 + psi_derivatives.p12
                   : psi_derivatives.m12 - psi_derivatives.p12) *
           0.5;
  }

  // Returns the (i,j)'s entry of Matrix B20.
  // @pre i = 0 or 1
  // @pre j = 0 or 1
  T B20(const PsiSigmaDerivatives& psi_derivatives, int i, int j) const {
    return (i == j ? psi_derivatives.m02 + psi_derivatives.p02
                   : psi_derivatives.m02 - psi_derivatives.p02) *
           0.5;
  }


  PsiSigmaDerivatives CalcPsiSigmaDerivative(const Matrix3<T>& FE) const;

  // Returns f(dev(τ)) = sqrt(3/2)‖ dev(τ) ‖ - τ_c, where dev(τ) is the deviatoric
  // component of Kirchhoff stress in the principal frame, and τ_c is yield_stress
  T ComputeYieldFunction(const Vector3<T>& deviatoric_tau) const;

  /**
   * If yields, apply ReturnMap(F_trial) to obtain the ELASTIC deformation
   * gradient FE (modify F_trial in place). 
   */
  void ApplyReturnMapping(
      const StrainStressData& strain_stress_data_trial, Matrix3<T>* F_trial) const;



  T yield_stress_;
  // sqrt(3.0/2.0)
  const double kSqrt3Over2 = 1.224744871391589;
  // used in clamping denominator.
  const double kEpsilon = 1e-12;  
};          

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
