#pragma once

#include <memory>

#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/mpm/internal/math_utils.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

/**
 * A base class providing the interface of constituive and plastic model.
 * Any elastoplastic model is parametrized by at least youngs_modulus E and
 * poissons_ratio nu.
 */
template <typename T>
class ElastoPlasticModel {
 public:
  // Default material, dough: E = 9e4 Pa, nu = 0.49
  ElastoPlasticModel();

  /**
   * Constructor uses Young's modulus E and Poisson's ratio nu
   * @pre youngs_modulus >= 0
   * @pre -1 < poissons_ratio < 0.5
   */
  ElastoPlasticModel(const T& youngs_modulus, const T& poissons_ratio);

  virtual std::unique_ptr<ElastoPlasticModel<T>> Clone() const = 0;

  T get_lambda() const { return lambda_; };
  T get_mu() const { return mu_; };
  T get_youngs_modulus() const { return youngs_modulus_; };
  T get_poissons_ratio() const { return poissons_ratio_; };

  /**
   * Resets youngs_modulus and updates mu and lambda accordingly, keeping
   * current poissons_ratio.
   * A temporary way to help with fracture!! Subject to change!!
   */
  void set_E(const T& youngs_modulus) {
    youngs_modulus_ = youngs_modulus;
    lambda_ = (youngs_modulus_ * poissons_ratio_ / (1 + poissons_ratio_) /
               (1 - 2 * poissons_ratio_));
    mu_ = (youngs_modulus_ / (2 * (1 + poissons_ratio_)));
  }

  /**
   * returns ψ(FE) where ψ is the energy density function. FE is the ELASTIC
   * part of deformation gradient.
   * TODO: consider adding an Eval function based on cached FE
   */
  virtual T CalcStrainEnergyDensity(const Matrix3<T>& FE) const = 0;

  /**
 * Computes the elastic deformation gradient FE from the trial deformation
gradient F_trial, and writes FE to F_trial. For elastic model, FE =
F_trial. For elastoplastic model, FE = ReturnMap(F_trial) when yield_stress
is violated.
* @pre F_trial != null_ptr
*/
  virtual void CalcFEFromFtrial(Matrix3<T>* F_trial) const = 0;

  /**
   * Calculates the first Piola Kirchhoff stress P = dψ(FE)/dFE
   * FE is the ELASTIC deformation gradient.
   * Remark: P = τ * FE^{-T}
   * @pre P != nullptr
   * TODO: consider adding an Eval function based on cached FE
   */
  virtual void CalcFirstPiolaStress(const Matrix3<T>& FE,
                                    Matrix3<T>* P) const = 0;
  /**
   * Calculates the Kirchhoff stress τ.
   * FE is the ELASTIC deformation gradient.
   * Remark: P = τ * FE^{-T}
   * @pre tau != nullptr
   * TODO: consider adding an Eval function based on cached FE
   */
  virtual void CalcKirchhoffStress(const Matrix3<T>& FE,
                                   Matrix3<T>* tau) const = 0;

  /* Calculates the derivative of first Piola stress with respect to the ELASTIC
 deformation gradient, given the ELASTIC deformation gradient FE. The stress
 derivative dPᵢⱼ/dFₖₗ is a 4-th order tensor that is flattened to a 9-by-9
 matrix. The 9-by-9 matrix is organized into 3-by-3 blocks of 3-by-3
 submatrices. The ik-th entry in the jl-th block corresponds to the value
 dPᵢⱼ/dFₖₗ. Let A denote the fourth order tensor dP/dF, then A is flattened to a
 9-by-9 matrix in the following way:

                     l = 1       l = 2       l = 3
                 -------------------------------------
                 |           |           |           |
       j = 1     |   Aᵢ₁ₖ₁   |   Aᵢ₁ₖ₂   |   Aᵢ₁ₖ₃   |
                 |           |           |           |
                 -------------------------------------
                 |           |           |           |
       j = 2     |   Aᵢ₂ₖ₁   |   Aᵢ₂ₖ₂   |   Aᵢ₂ₖ₃   |
                 |           |           |           |
                 -------------------------------------
                 |           |           |           |
       j = 3     |   Aᵢ₃ₖ₁   |   Aᵢ₃ₖ₂   |   Aᵢ₃ₖ₃   |
                 |           |           |           |
                 -------------------------------------
  For instance, the first row (1by9) of the matrix is
  [dP₁₁dF₁₁, dP₁₁dF₂₁, dP₁₁dF₃₁, dP₁₁dF₁₂, dP₁₁dF₂₂, dP₁₁dF₃₂, dP₁₁dF₁₃,
 dP₁₁dF₂₃, dP₁₁dF₃₃] Both the numerator (3by3 matrix) and denominator (3by3
 matrix) are flattened column-wise. dPᵢⱼ/dFₖₗ = dPdF(i+3*j, k+3*l), assuming
 indices are 0,1,2 rather than 1,2,3.
 * @pre dPdF != nullptr
 * Remark: This should be a symmetric matrix!!
 */
  virtual void CalcFirstPiolaStressDerivative(
      const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const = 0;

  /**
   * TODO: This should not be deprecated!! Remove this after mpm_transfer.h is
   * finished
   */
  virtual void UpdateDeformationGradientAndCalcKirchhoffStress(
      Matrix3<T>* tau, Matrix3<T>* elastic_deformation_gradient) const = 0;

  // returns det(FE)-1
  T ComputeJMinusOne(const Matrix3<T>& FE) const {
    return FE.determinant() - 1.0;
  }

  /**
   * Computes the polar decomposition FE = R * S
   * @pre R != nullptr
   * @pre S != nullptr
   */
  void ComputePolarDecomp(const Matrix3<T>& FE, Matrix3<T>* R,
                          Matrix3<T>* S) const {
    fem::internal::PolarDecompose<T>(FE, R, S);
  }

  /**
   * Computes det(FE) * FE^T
   * @pre JFinvT != nullptr
   */
  void ComputeJFinvT(const Matrix3<T>& FE, Matrix3<T>* JFinvT) const {
    fem::internal::CalcCofactorMatrix<T>(FE, JFinvT);
  }

  virtual ~ElastoPlasticModel() = default;

 protected:
  T lambda_{0};  // The first Lamé coefficient
  T mu_{0};      // The second Lamé coefficient
  T youngs_modulus_{0};
  T poissons_ratio_{0};
};  // class ElastoPlasticModel

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
