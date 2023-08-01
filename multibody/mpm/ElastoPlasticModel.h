#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/common/autodiff.h"
#include "drake/multibody/mpm/MathUtils.h"
#include <iostream>

namespace drake {
namespace multibody {
namespace mpm {

// A base class providing the interface of constituive and plastic model
template <typename T>
class ElastoPlasticModel {
 public:
    // Default material, dough: E = 9e4 Pa, nu = 0.49
    ElastoPlasticModel();

    // Constructor uses Young's modulus E and Poisson's ratio nu
    ElastoPlasticModel(T E, T nu);

    virtual std::unique_ptr<ElastoPlasticModel<T>> Clone() const = 0;

    T get_lambda() const {  return lambda_;  };
    T get_mu() const     {  return mu_;  };

    virtual T CalcStrainEnergyDensity(const Matrix3<T>& FE) const = 0;

    virtual void CalcFirstPiolaStress(const Matrix3<T>& FE, Matrix3<T>* P) const = 0;


    /* Calculates the derivative of first Piola stress with respect to the
   deformation gradient, given the deformation gradient related quantities
   contained in `data`. The stress derivative dPᵢⱼ/dFₖₗ is a 4-th order tensor
   that is flattened to a 9-by-9 matrix. The 9-by-9 matrix is organized into
   3-by-3 blocks of 3-by-3 submatrices. The ik-th entry in the jl-th block
   corresponds to the value dPᵢⱼ/dFₖₗ. Let A denote the fourth order tensor
   dP/dF, then A is flattened to a 9-by-9 matrix in the following way:

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
    [dP₁₁dF₁₁, dP₁₁dF₂₁, dP₁₁dF₃₁, dP₁₁dF₁₂, dP₁₁dF₂₂, dP₁₁dF₃₂, dP₁₁dF₁₃, dP₁₁dF₂₃, dP₁₁dF₃₃] 
    Both the numerator (3by3 matrix) and denominator (3by3 matrix) are flattened column-wise.
     dPᵢⱼ/dFₖₗ = dPdF(i+3*j, k+3*l), assuming indices are 0,1,2 rather than 1,2,3.
    Remark: This should be and is a symmetric matrix!!
   */
    virtual void CalcFirstPiolaStressDerivative(const Matrix3<T>& FE, Eigen::Matrix<T, 9, 9>* dPdF) const = 0;

    T dummy_function(const Matrix3<T>& FE) const {
      return FE(0,0) * FE(0,0) + FE(0,1) * FE(0,1) + 3.0 * FE(0,2);
    }

    void dummy_function_derivative(const Matrix3<T>& FE, Matrix3<T>* derivative) const {
      (*derivative)(0,0) = (2.0) * FE(0,0);
      (*derivative)(0,1) = (2.0) * FE(0,1);
      (*derivative)(0,2) = 3.0;
      (*derivative)(1,0) = 0.0;
      (*derivative)(1,1) = 0.0;
      (*derivative)(1,2) = 0.0;
      (*derivative)(2,0) = 0.0;
      (*derivative)(2,1) = 0.0;
      (*derivative)(2,2) = 0.0;
    }

    T dummy_function2(const Eigen::Matrix3X<T>& Xp){

      Eigen::MatrixX<T> result = Xp.array().square();
      return result.sum();
    }

    void dummy_function2_derivative(const Eigen::Matrix3X<T>& Xp, Eigen::Matrix3X<T>* derivative) const {
      (*derivative) = 2 * Xp;
      // (*derivative)(0,0) = (*derivative)(0,0) + 0.1;
    }

    void TestAssignF(Eigen::Matrix3X<T>& Xp, std::vector<Matrix3<T>>* F_from_Xp){
        int kParticles = Xp.cols();
        (F_from_Xp)->clear();
        for (int i = 0; i < kParticles; i++) {
            Eigen::Vector3<T> m = Xp.col(i);
            m(1) = 2.0 * m(1);
            m(2) = 3.0 * m(2);
            F_from_Xp->push_back(m.asDiagonal());
        }
    }

    void dummy_assign_F(const Eigen::Vector3<T>& X, Eigen::Matrix3<T>* F_from_Xp) {
      (*F_from_Xp)(0,0) = X(0);
      (*F_from_Xp)(0,1) = 0.0;
      (*F_from_Xp)(0,2) = 0.0;
      (*F_from_Xp)(1,0) = 0.0;
      (*F_from_Xp)(1,1) = 2.0 * X(1);
      (*F_from_Xp)(1,2) = 0.0;
      (*F_from_Xp)(2,0) = 0.0;
      (*F_from_Xp)(2,1) = 0.0;
      (*F_from_Xp)(2,2) = 3.0 * X(2);
    } 

    T dummy_psi_of_F(const Eigen::Matrix3X<T>& F){
      return F.sum();
    }


    // Update the elastic deformation gradient according to the plasticity model
    // by projecting the trial elastic stress to the yield surface. Then
    // calculate the projected Kirchhoff stress by the projected deformation
    // gradient. Kirchhoff stress density is defined as tau = P Fᴱ^T
    //                                                      = dψ/dFᴱ Fᴱ^T,
    // where ψ denotes the energy density
    virtual void UpdateDeformationGradientAndCalcKirchhoffStress(
                    Matrix3<T>* tau,
                    Matrix3<T>* elastic_deformation_gradient) const = 0;


    T computeJm1(const Matrix3<T>& FE) const {
      return FE.determinant() - 1.0;
    }
    void computePolarDecomp(const Matrix3<T>& FE, Matrix3<T>* R, Matrix3<T>* S) const {
      fem::internal::PolarDecompose<T>(FE, R, S);
    }
    void computeJFinvT(const Matrix3<T>& FE, Matrix3<T>* JFinvT) const {
      fem::internal::CalcCofactorMatrix<T>(FE, JFinvT);
    }

    virtual ~ElastoPlasticModel() = default;

 protected:
    T lambda_;                       // The first Lamé coefficient
    T mu_;                           // The second Lamé coefficient
};  // class ElastoPlasticModel

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
