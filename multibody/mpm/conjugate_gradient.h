#pragma once

#include <iostream>
#include <utility>

#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {

struct HessianWrapper {
  HessianWrapper(const MpmTransfer<double>& transfer_in,
                 const MpmModel<double>& model_in,
                 const DeformationState<double>& deformation_state_in,
                 double dt_in)
      : transfer(transfer_in),
        model(model_in),
        deformation_state(deformation_state_in),
        dt(dt_in) {}

  // result = hessian * z;
  void Multiply(const Eigen::VectorXd& z, Eigen::VectorXd* result) const {
    result->resizeLike(z);
    result->setZero();
    model.AddD2EnergyDV2TimesZ(z, transfer, deformation_state, dt, result);
  }

  // assuming there is a precondition matrix M, solver for Mx = rhs
  void Precondition(const Eigen::VectorXd& rhs, Eigen::VectorXd* x) const {
    model.Precondition(deformation_state, rhs, x);
  }

  const MpmTransfer<double>& transfer;
  const MpmModel<double>& model;
  const DeformationState<double>& deformation_state;
  double dt;
};

// solve for Ax=b
// A must provide precondition() and multiply(z) = A*z
class ConjugateGradient {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConjugateGradient);

  ConjugateGradient() {}

  void SetRelativeTolerance(double tol) { relative_tolerance_ = tol; }
  void SetAbsoluteTolerance(double tol) { absolute_tolerance_ = tol; }

  // model and state together give A
  // x is modified in place
  // CG initial guess x = 0
  int Solve(const HessianWrapper& A, const Eigen::VectorXd& b,
            Eigen::VectorXd* x) {
    DRAKE_DEMAND(x != nullptr);
    x->resizeLike(b);
    x->setZero();

    r_ = b;  // r = b - Ax = b - 0 = b;
    z_.resizeLike(b);
    A.Precondition(r_, &z_);  // find z_: Mz_ = r_
    p_ = z_;
    double b_norm = b.norm();
    double rkTzk = r_.dot(z_);
    int k = 0;
    for (; k < max_CG_iter_; ++k) {
      if (r_.norm() <
          std::max(relative_tolerance_ * b_norm, absolute_tolerance_)) {
        break;
      }

      A.Multiply(p_, &Ap_);

      double alpha = rkTzk / p_.dot(Ap_);

      (*x) += alpha * p_;

      r_ -= alpha * Ap_;

      A.Precondition(r_, &z_);  // find z_: Mz_ = r_

      double rkp1Tzkp1 = r_.dot(z_);

      double beta = rkp1Tzkp1 / rkTzk;

      p_ = z_ + beta * p_;

      rkTzk = rkp1Tzkp1;
    }

    std::cout << "CG converged after " << k
              << " iterations, residual = " << r_.norm() << std::endl;
    return k;
  }

  Eigen::VectorXd r_;
  Eigen::VectorXd z_;
  Eigen::VectorXd p_;
  Eigen::VectorXd Ap_;  // stores A*p_

  double relative_tolerance_ = 1e-5;
  double absolute_tolerance_ = 1e-6;
  int max_CG_iter_ = 1e5; 
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
