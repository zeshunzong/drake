#pragma once

#include <utility>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// A implementation of 3D quadratic B spline
/**
 * An implementation of 3D quadratic B spline.
 * This gives an (interpolation) function centered at position with support
 * being [-1.5h,1.5h] in each dimension See
 * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf page
 * 32-34
 */
template <typename T>
class BSpline {
 public:
  BSpline(const T h, const Vector3<T>& position);

  /**
   * Default to h = 1.0 and position = [0,0,0]
   */
  BSpline();

  T get_h() const { return h_; }
  Vector3<T> get_position() { return position_; }

  /**
   * Checks whether a point with position x is inside the support of this
   * Bspline x is in support if |x(i)-center(i)|<1.5h for i = 0, 1, 2
   */
  bool InSupport(const Vector3<T>& x) const;

  // Evaluation of Bspline basis and gradient on a particular position x
  T EvalBasis(const Vector3<T>& x) const;
  Vector3<T> EvalGradientBasis(const Vector3<T>& x) const;
  std::pair<T, Vector3<T>> EvalBasisAndGradient(const Vector3<T>& x) const;

 private:
  // Helper function. Evaluate the values and the gradients of 1D quadratic
  // Bspline on the reference 1D domain r, note that the basis has compact
  // support in [-1.5, 1.5]
  T Eval1DBasis(T r) const;
  T EvalGradient1DBasis(T r) const;

  T h_{};                  // The scaling of the reference domain.
                           // Since the class is for the usage of
                           // MPM, we assume we are on the uniform
                           // grid, and the physical domain (x, y,
                           // z) can be transformed to the
                           // reference domain (r, s, t) through a
                           // affine transformation \phi^-1(x, y,
                           // z) = (r, s, t) = 1/h*(x-xc, y-yc,
                           // z-zc). The basis on physical domain
                           // will have support [-1.5h, 1.5h]
  Vector3<T> position_{};  // The "center" of the Bspline, or (xc,
                           // yc, zc) in above comment
};                         // class BSpline

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
