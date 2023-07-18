#include "drake/multibody/mpm/StvkHenckyWithVonMisesModel.h"

namespace drake {
namespace multibody {
namespace mpm {

StvkHenckyWithVonMisesModel::StvkHenckyWithVonMisesModel(double tau_c):
                                    ElastoPlasticModel(), yield_stress_(tau_c) {
    DRAKE_ASSERT(tau_c > 0);
}

StvkHenckyWithVonMisesModel::StvkHenckyWithVonMisesModel(double E, double nu,
                                                         double tau_c):
                                                    ElastoPlasticModel(E, nu),
                                                    yield_stress_(tau_c) {
    DRAKE_ASSERT(tau_c > 0);
}

double StvkHenckyWithVonMisesModel::EvalYieldFunction(const Matrix3<double>& FE)
                                                                        const {
    Eigen::JacobiSVD<Matrix3<double>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    StrainData trial_strain_data = CalcStrainData(FE);
    StressData trial_stress_data = CalcStressData(trial_strain_data);
    return EvalYieldFunction(trial_stress_data);
}

// TODO(yiminlin.tri): do nothing
double StvkHenckyWithVonMisesModel::CalcStrainEnergyDensity(const Matrix3<double>&) const {  return 0.0; };

void StvkHenckyWithVonMisesModel::
                                UpdateDeformationGradientAndCalcKirchhoffStress(
                        Matrix3<double>* tau, Matrix3<double>* FE_trial) const {
    Eigen::JacobiSVD<Matrix3<double>>
                    svd(*FE_trial, Eigen::ComputeFullU | Eigen::ComputeFullV);
    StrainData trial_strain_data = CalcStrainData(*FE_trial);
    StressData trial_stress_data = CalcStressData(trial_strain_data);
    ProjectDeformationGradientToYieldSurface(trial_stress_data,
                                            &trial_strain_data, FE_trial);
    CalcKirchhoffStress(trial_strain_data, tau);
}

StvkHenckyWithVonMisesModel::StrainData
        StvkHenckyWithVonMisesModel::CalcStrainData(const Matrix3<double>& FE)
                                                                        const {
    Eigen::JacobiSVD<Matrix3<double>>
                    svd(FE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3<double>& U = svd.matrixU();
    const Matrix3<double>& V = svd.matrixV();
    Vector3<double> sigma    = svd.singularValues();
    // Since the Hencky strain has form ε = 1/2ln(Fᴱ Fᴱ^T)
    // the Hencky strain in the principal frame is ln(Σ), where Fᴱ = U Σ Vᵀ
    Vector3<double> eps_hat = sigma.array().log();
    // trace of the trial Hencky strain tr(ε)
    double tr_eps           = eps_hat.sum();

    return {U, V, eps_hat, tr_eps};
}

StvkHenckyWithVonMisesModel::StressData
        StvkHenckyWithVonMisesModel::CalcStressData(
                const StvkHenckyWithVonMisesModel::StrainData& strain_data)
                                                                        const {
    // Vector of trace of the trial stress
    Vector3<double> tr_eps_vec = strain_data.tr_eps*Vector3<double>::Ones();
    // The deviatoric component of Kirchhoff stress in the principal frame is:
    // dev(τ) = τ - pJI = τ - 1/3tr(τ)I
    // In the principal frame: dev(τ) = 2μlog(σᵢ) - 2μ/3 ∑ᵢ log(σᵢ)
    Vector3<double> tau_dev_hat =
                        2*mu_*(strain_data.eps_hat-1.0/3.0*tr_eps_vec);
    double tau_dev_hat_norm = tau_dev_hat.norm();

    return {tau_dev_hat, tau_dev_hat_norm};
}

void StvkHenckyWithVonMisesModel::CalcKirchhoffStress(
            const StvkHenckyWithVonMisesModel::StrainData& trial_strain_data,
                                                Matrix3<double>* tau) const {
    // Calculate the singular values of Kirchhoff stress
    // Σᵢ = 2μ log(σᵢ) + λ ∑ᵢ log(σᵢ)
    Vector3<double> sigma_tau;
    for (int d = 0; d < 3; ++d) {
        sigma_tau(d)  = 2*mu_*trial_strain_data.eps_hat(d)
                      + lambda_*trial_strain_data.tr_eps;
    }
    // The Kirchhoff stress can be then written as
    // τ = U Σᵢ Uᵀ, where U are left singular vectors of the elastic deformation
    //             gradient FE
    const Matrix3<double>& U = trial_strain_data.U;
    *tau = U * sigma_tau.asDiagonal() * U.transpose();
}

double StvkHenckyWithVonMisesModel::EvalYieldFunction(const StressData&
                                                      trial_stress_data) const {
    return StvkHenckyWithVonMisesModel::sqrt_32
          *trial_stress_data.tau_dev_hat_norm - yield_stress_;
}

void StvkHenckyWithVonMisesModel::
        ProjectDeformationGradientToYieldSurface(
            const StvkHenckyWithVonMisesModel::StressData& trial_stress_data,
            StvkHenckyWithVonMisesModel::StrainData* trial_strain_data,
            Matrix3<double>* FE_trial) const {
    // If the trial stress τ is in the yield surface f(τ) <= 0, plasticity is
    // not applied.
    // Otherwise, project the trial stress τ in the plastic flow direction
    // df/dτ τ onto the yield surface f(τ) <= 0
    // The trial stress is on the yield surface, f(τ) <= 0, if and only if the
    // singular values of the deviatoric component of trial stress satisfies:
    // f(τ) = sqrt(3/2)‖ dev(τ) ‖ - τ_c ≤ 0
    double f_tau = EvalYieldFunction(trial_stress_data);
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
        Vector3<double> nu = StvkHenckyWithVonMisesModel::sqrt_32
                            *trial_stress_data.tau_dev_hat
                            /trial_stress_data.tau_dev_hat_norm;
        double delta_gamma = f_tau/(3.0*mu_);
        // Update the singular values of Hencky strain ε
        trial_strain_data->eps_hat =
                                trial_strain_data->eps_hat - delta_gamma*nu;
        // F_proj = exp(ε_proj) in the principal frame
        Vector3<double> proj_F_hat = (trial_strain_data->eps_hat).array().exp();
        // New elastic deformation gradient projected to the yield surface
        // proj(Fᴱ)
        *FE_trial = trial_strain_data->U
                   *proj_F_hat.asDiagonal()
                   *trial_strain_data->V.transpose();
    }
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
