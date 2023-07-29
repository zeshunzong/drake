#pragma once

#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/common/autodiff.h"
#include <iostream>
namespace drake {
namespace multibody {
namespace mpm {

// A implementation of 3D quadratic B spline
template <typename T>
class BSpline {
 public:
    BSpline();
    BSpline(const T h, const Vector3<T>& pos);

    // Check whether the input point is inside the support of this Bspline
    bool InSupport(const Vector3<T>& x) const;

    // Evaluation of Bspline basis on a particular position
    T EvalBasis(const Vector3<T>& x) const;
    // TODO(yiminlin.tri): Pass in pointer to avoid allocations
    Vector3<T> EvalGradientBasis(const Vector3<T>& x) const;
    std::pair<T, Vector3<T>>
      EvalBasisAndGradient(const Vector3<T> & x) const;

    // Helper function
    T get_h() const;
    Vector3<T> get_position() const;

 private:
    // Helper function. Evaluate the values and the gradients of 1D quadratic
    // Bspline on the reference 1D domain r, note that the basis has compact
    // support in [-1.5, 1.5]
    T Eval1DBasis(T r) const;
    T EvalGradient1DBasis(T r) const;

    T h_{};                        // The scaling of the reference domain.
                                        // Since the class is for the usage of
                                        // MPM, we assume we are on the uniform
                                        // grid, and the physical domain (x, y,
                                        // z) can be transformed to the
                                        // reference domain (r, s, t) through a
                                        // affine transformation \phi^-1(x, y,
                                        // z) = (r, s, t) = 1/h*(x-xc, y-yc,
                                        // z-zc). The basis on physical domain
                                        // will have support [-1.5h, 1.5h]
    Vector3<T> position_{};        // The "center" of the Bspline, or (xc,
                                        // yc, zc) in above comment
};  // class BSpline

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
