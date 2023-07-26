#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
StvkHenckyWithVonMisesModel<T>::StvkHenckyWithVonMisesModel(T tau_c):
                                    ElastoPlasticModel<T>(), yield_stress_(tau_c) {
    DRAKE_ASSERT(tau_c > 0);
}

template <typename T>
StvkHenckyWithVonMisesModel<T>::StvkHenckyWithVonMisesModel(T E, T nu,
                                                         T tau_c):
                                                    ElastoPlasticModel<T>(E, nu),
                                                    yield_stress_(tau_c) {
    DRAKE_ASSERT(tau_c > 0);
}
template <typename T>
T StvkHenckyWithVonMisesModel<T>::EvalYieldFunction(const Matrix3<T>& FE)
                                                                        const {
    Eigen::JacobiSVD<Matrix3<T>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    StrainData trial_strain_data = CalcStrainData(FE);
    StressData trial_stress_data = CalcStressData(trial_strain_data);
    return EvalYieldFunction(trial_stress_data);
}

template <typename T>
T StvkHenckyWithVonMisesModel<T>::CalcStrainEnergyDensity(const Matrix3<T>& FE) const {  
    //psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2

    StrainData strain_data = CalcStrainData(FE);
    return this->mu_ * (strain_data.eps_hat(0)*strain_data.eps_hat(0) +
                         strain_data.eps_hat(1)*strain_data.eps_hat(1) +
                         strain_data.eps_hat(2)*strain_data.eps_hat(2)) +
                         0.5 * this->lambda_ * (strain_data.eps_hat(0)+strain_data.eps_hat(1)+strain_data.eps_hat(2))
                          * (strain_data.eps_hat(0)+strain_data.eps_hat(1)+strain_data.eps_hat(2));
}

// template <typename T>
// void StvkHenckyWithVonMisesModel<T>::CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const {
//     // P = U (2 mu S^{-1} (log S) + lambda tr(log S) S^{-1}) V^T
//     StrainData strain_data = CalcStrainData(FE);

//     Vector3<T> P_hat = 2 * this->mu_ * strain_data.eps_hat + this->lambda_ * strain_data.tr_eps*Vector3<T>::Ones();
//     P_hat = strain_data.sigma_inverse.asDiagonal() * P_hat;
//     (*P) = strain_data.U * P_hat.asDiagonal() * strain_data.V.transpose();
// }

// template <typename T>
// void StvkHenckyWithVonMisesModel<T>::CalcFirstPiolaStressDerivative(const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const {
//     PsiSigmaDerivative psi_derivatives = CalcPsiSigmaDerivative(FE);
//     StrainData strain_data = CalcStrainData(FE);
//     for (int ij = 0; ij < 9; ++ij) {
//         int j = ij / 3;
//         int i = ij - j * 3;
//         for (int rs = 0; rs <= ij; ++rs) {
//             int s = rs / 3;
//             int r = rs - s * 3;
//             (*dPdF)(ij, rs) = (*dPdF)(rs, ij) = psi_derivatives.psi00 * strain_data.U(i, 0) * strain_data.V(j, 0) * strain_data.U(r, 0) * strain_data.V(s, 0)
//                                                 + psi_derivatives.psi01 * strain_data.U(i, 0) * strain_data.V(j, 0) * strain_data.U(r, 1) * strain_data.V(s, 1) 
//                                                 + psi_derivatives.psi02 * strain_data.U(i, 0) * strain_data.V(j, 0) * strain_data.U(r, 2) * strain_data.V(s, 2) 
//                                                 + psi_derivatives.psi01 * strain_data.U(i, 1) * strain_data.V(j, 1) * strain_data.U(r, 0) * strain_data.V(s, 0) 
//                                                 + psi_derivatives.psi11 * strain_data.U(i, 1) * strain_data.V(j, 1) * strain_data.U(r, 1) * strain_data.V(s, 1) 
//                                                 + psi_derivatives.psi12 * strain_data.U(i, 1) * strain_data.V(j, 1) * strain_data.U(r, 2) * strain_data.V(s, 2) 
//                                                 + psi_derivatives.psi02 * strain_data.U(i, 2) * strain_data.V(j, 2) * strain_data.U(r, 0) * strain_data.V(s, 0) 
//                                                 + psi_derivatives.psi12 * strain_data.U(i, 2) * strain_data.V(j, 2) * strain_data.U(r, 1) * strain_data.V(s, 1) 
//                                                 + psi_derivatives.psi22 * strain_data.U(i, 2) * strain_data.V(j, 2) * strain_data.U(r, 2) * strain_data.V(s, 2) 
//                                                 + b01(psi_derivatives, 0, 0) * strain_data.U(i, 0) * strain_data.V(j, 1) * strain_data.U(r, 0) * strain_data.V(s, 1) 
//                                                 + b01(psi_derivatives, 0, 1) * strain_data.U(i, 0) * strain_data.V(j, 1) * strain_data.U(r, 1) * strain_data.V(s, 0) 
//                                                 + b01(psi_derivatives, 1, 0) * strain_data.U(i, 1) * strain_data.V(j, 0) * strain_data.U(r, 0) * strain_data.V(s, 1) 
//                                                 + b01(psi_derivatives, 1, 1) * strain_data.U(i, 1) * strain_data.V(j, 0) * strain_data.U(r, 1) * strain_data.V(s, 0) 
//                                                 + b12(psi_derivatives, 0, 0) * strain_data.U(i, 1) * strain_data.V(j, 2) * strain_data.U(r, 1) * strain_data.V(s, 2) 
//                                                 + b12(psi_derivatives, 0, 1) * strain_data.U(i, 1) * strain_data.V(j, 2) * strain_data.U(r, 2) * strain_data.V(s, 1) 
//                                                 + b12(psi_derivatives, 1, 0) * strain_data.U(i, 2) * strain_data.V(j, 1) * strain_data.U(r, 1) * strain_data.V(s, 2) 
//                                                 + b12(psi_derivatives, 1, 1) * strain_data.U(i, 2) * strain_data.V(j, 1) * strain_data.U(r, 2) * strain_data.V(s, 1) 
//                                                 + b20(psi_derivatives, 1, 1) * strain_data.U(i, 0) * strain_data.V(j, 2) * strain_data.U(r, 0) * strain_data.V(s, 2) 
//                                                 + b20(psi_derivatives, 1, 0) * strain_data.U(i, 0) * strain_data.V(j, 2) * strain_data.U(r, 2) * strain_data.V(s, 0) 
//                                                 + b20(psi_derivatives, 0, 1) * strain_data.U(i, 2) * strain_data.V(j, 0) * strain_data.U(r, 0) * strain_data.V(s, 2) 
//                                                 + b20(psi_derivatives, 0, 0) * strain_data.U(i, 2) * strain_data.V(j, 0) * strain_data.U(r, 2) * strain_data.V(s, 0);
//         }
//     }

// }

template <typename T>
void StvkHenckyWithVonMisesModel<T>::
                                UpdateDeformationGradientAndCalcKirchhoffStress(
                        Matrix3<T>* tau, Matrix3<T>* FE_trial) const {
    Eigen::JacobiSVD<Matrix3<T>>
                    svd(*FE_trial, Eigen::ComputeFullU | Eigen::ComputeFullV);
    StrainData trial_strain_data = CalcStrainData(*FE_trial);
    StressData trial_stress_data = CalcStressData(trial_strain_data);
    ProjectDeformationGradientToYieldSurface(trial_stress_data,
                                            &trial_strain_data, FE_trial);
    CalcKirchhoffStress(trial_strain_data, tau);
}

template <typename T>
StvkHenckyWithVonMisesModel<T>::StrainData
        StvkHenckyWithVonMisesModel<T>::CalcStrainData(const Matrix3<T>& FE)
                                                                        const {
    Eigen::JacobiSVD<Matrix3<T>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3<T>& U = svd.matrixU();
    const Matrix3<T>& V = svd.matrixV();
    Vector3<T> sigma    = svd.singularValues();
    // Since the Hencky strain has form ε = 1/2ln(Fᴱ Fᴱ^T)
    // the Hencky strain in the principal frame is ln(Σ), where Fᴱ = U Σ Vᵀ
    Vector3<T> eps_hat = sigma.array().log();
    // trace of the trial Hencky strain tr(ε)
    T tr_eps           = eps_hat.sum();
    Vector3<T> sigma_inverse {1/sigma(0), 1/sigma(1), 1/sigma(2)};

    return {U, V, eps_hat, tr_eps, sigma, sigma_inverse};
}

template <typename T>
StvkHenckyWithVonMisesModel<T>::StressData
        StvkHenckyWithVonMisesModel<T>::CalcStressData(
                const StvkHenckyWithVonMisesModel<T>::StrainData& strain_data)
                                                                        const {
    // Vector of trace of the trial stress
    Vector3<T> tr_eps_vec = strain_data.tr_eps*Vector3<T>::Ones();
    // The deviatoric component of Kirchhoff stress in the principal frame is:
    // dev(τ) = τ - pJI = τ - 1/3tr(τ)I
    // In the principal frame: dev(τ) = 2μlog(σᵢ) - 2μ/3 ∑ᵢ log(σᵢ)
    Vector3<T> tau_dev_hat =
                        2*ElastoPlasticModel<T>::mu_*(strain_data.eps_hat-1.0/3.0*tr_eps_vec);
    T tau_dev_hat_norm = tau_dev_hat.norm();

    return {tau_dev_hat, tau_dev_hat_norm};
}

// template <typename T>
// StvkHenckyWithVonMisesModel<T>::PsiSigmaDerivative 
//         StvkHenckyWithVonMisesModel<T>::CalcPsiSigmaDerivative(const Matrix3<T>& FE) const {
//     StrainData strain_data = CalcStrainData(FE);
//     T twoMuPlusLam = 2 * this->mu_ + this->lambda_;
//     // dpsi_dsigma
//     T psi0 = (this->mu_ * 2 * strain_data.eps_hat(0) + this->lambda_ * strain_data.tr_eps) / strain_data.sigma(0);
//     T psi1 = (this->mu_ * 2 * strain_data.eps_hat(1) + this->lambda_ * strain_data.tr_eps) / strain_data.sigma(1);
//     T psi2 = (this->mu_ * 2 * strain_data.eps_hat(2) + this->lambda_ * strain_data.tr_eps) / strain_data.sigma(2);
//     // (psiA-psiB)/(sigmaA-sigmaB)
//     using std::abs;
//     T m01 = -(this->lambda_ * strain_data.tr_eps + 
//                                     this->mu_ * 2 * mathutils::diff_interlock_log_over_diff(strain_data.sigma(0), abs(strain_data.sigma(1)), strain_data.eps_hat(1), epsilon_threshold_)) / (strain_data.sigma(0) * strain_data.sigma(1));
//     T m02 = -(this->lambda_ * strain_data.tr_eps + 
//                                     this->mu_ * 2 * mathutils::diff_interlock_log_over_diff(strain_data.sigma(0), abs(strain_data.sigma(2)), strain_data.eps_hat(2), epsilon_threshold_)) / (strain_data.sigma(0) * strain_data.sigma(2));
//     T m12 = -(this->lambda_ * strain_data.tr_eps + 
//                                     this->mu_ * 2 * mathutils::diff_interlock_log_over_diff(strain_data.sigma(1), abs(strain_data.sigma(2)), strain_data.eps_hat(2), epsilon_threshold_)) / (strain_data.sigma(1) * strain_data.sigma(2));
//     // (psiA+psiB)/(sigmaA+sigmaB)
//     T p01 = (psi0 + psi1) / mathutils::clamp_small_magnitude(strain_data.sigma(0) + strain_data.sigma(1), epsilon_threshold_);
//     T p02 = (psi0 + psi2) / mathutils::clamp_small_magnitude(strain_data.sigma(0) + strain_data.sigma(2), epsilon_threshold_);
//     T p12 = (psi1 + psi2) / mathutils::clamp_small_magnitude(strain_data.sigma(1) + strain_data.sigma(2), epsilon_threshold_);
//     T inv2_0 = 1 / (strain_data.sigma(0) * strain_data.sigma(0));
//     T inv2_1 = 1 / (strain_data.sigma(1) * strain_data.sigma(1));
//     T inv2_2 = 1 / (strain_data.sigma(2) * strain_data.sigma(2));
//     // d2psi_dsigma2
//     T psi00 = (twoMuPlusLam * (1 - strain_data.eps_hat(0)) - this->lambda_ * (strain_data.eps_hat(1) + strain_data.eps_hat(2))) * inv2_0;
//     T psi11 = (twoMuPlusLam * (1 - strain_data.eps_hat(1)) - this->lambda_ * (strain_data.eps_hat(0) + strain_data.eps_hat(2))) * inv2_1;
//     T psi22 = (twoMuPlusLam * (1 - strain_data.eps_hat(2)) - this->lambda_ * (strain_data.eps_hat(0) + strain_data.eps_hat(1))) * inv2_2;
//     T psi01 = this->lambda_ / strain_data.sigma(0) / strain_data.sigma(1);
//     T psi12 = this->lambda_ / strain_data.sigma(1) / strain_data.sigma(2);
//     T psi02 = this->lambda_ / strain_data.sigma(2) / strain_data.sigma(0);
//     return {psi0, psi1, psi2, psi00, psi11, psi22, psi01, psi02, psi12, m01, p01, m02, p02, m12, p12};
// }

template <typename T>
void StvkHenckyWithVonMisesModel<T>::CalcKirchhoffStress(
            const StvkHenckyWithVonMisesModel<T>::StrainData& trial_strain_data,
                                                Matrix3<T>* tau) const {
    // Calculate the singular values of Kirchhoff stress
    // Σᵢ = 2μ log(σᵢ) + λ ∑ᵢ log(σᵢ)
    Vector3<T> sigma_tau;
    for (int d = 0; d < 3; ++d) {
        sigma_tau(d)  = 2*ElastoPlasticModel<T>::mu_*trial_strain_data.eps_hat(d)
                      + ElastoPlasticModel<T>::lambda_*trial_strain_data.tr_eps;
    }
    // The Kirchhoff stress can be then written as
    // τ = U Σᵢ Uᵀ, where U are left singular vectors of the elastic deformation
    //             gradient FE
    const Matrix3<T>& U = trial_strain_data.U;
    *tau = U * sigma_tau.asDiagonal() * U.transpose();
}

template <typename T>
T StvkHenckyWithVonMisesModel<T>::EvalYieldFunction(const StressData&
                                                      trial_stress_data) const {
    return StvkHenckyWithVonMisesModel<T>::sqrt_32
          *trial_stress_data.tau_dev_hat_norm - yield_stress_;
}

template <typename T>
void StvkHenckyWithVonMisesModel<T>::
        ProjectDeformationGradientToYieldSurface(
            const StvkHenckyWithVonMisesModel<T>::StressData& trial_stress_data,
            StvkHenckyWithVonMisesModel<T>::StrainData* trial_strain_data,
            Matrix3<T>* FE_trial) const {
    // If the trial stress τ is in the yield surface f(τ) <= 0, plasticity is
    // not applied.
    // Otherwise, project the trial stress τ in the plastic flow direction
    // df/dτ τ onto the yield surface f(τ) <= 0
    // The trial stress is on the yield surface, f(τ) <= 0, if and only if the
    // singular values of the deviatoric component of trial stress satisfies:
    // f(τ) = sqrt(3/2)‖ dev(τ) ‖ - τ_c ≤ 0
    T f_tau = EvalYieldFunction(trial_stress_data);
    bool in_yield_surface =  f_tau <= 0.0;
    if (!in_yield_surface) {
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
        // Since f(τⁿ⁺¹) = ‖ τⁿ⁺¹ ‖ - τ_c = 0, i.e. τⁿ⁺¹ is on the yield surface
        // Δγ = f(τ)/(3μ)
        // By the definition of Hencky strain,
        Vector3<T> nu = StvkHenckyWithVonMisesModel::sqrt_32
                            *trial_stress_data.tau_dev_hat
                            /trial_stress_data.tau_dev_hat_norm;
        T delta_gamma = f_tau/(3.0*ElastoPlasticModel<T>::mu_);
        // Update the singular values of Hencky strain ε
        trial_strain_data->eps_hat =
                                trial_strain_data->eps_hat - delta_gamma*nu;
        // F_proj = exp(ε_proj) in the principal frame
        Vector3<T> proj_F_hat = (trial_strain_data->eps_hat).array().exp();
        // New elastic deformation gradient projected to the yield surface
        // proj(Fᴱ)
        *FE_trial = trial_strain_data->U
                   *proj_F_hat.asDiagonal()
                   *trial_strain_data->V.transpose();
    }
}
template class StvkHenckyWithVonMisesModel<AutoDiffXd>;
template class StvkHenckyWithVonMisesModel<double>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
